import math
import taichi as ti
from utilities import execute_command
from utilities import create_directory
from utilities import write_to_text_file

real = ti.f32
ti.set_default_fp(real)
max_steps = 4096
steps = 50
assert(steps*2 <= max_steps)

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(3, dt=real)
mat = lambda: ti.Matrix(3, 3, dt=real)

loss = scalar()
x = vec()
v = vec()
rotation = vec()
omega = vec()
friction_x = scalar()
friction_z = scalar()
halfsize = vec()
inverse_mass = scalar()
inverse_inertia = mat()
v_inc = vec()
omega_inc = vec()
J_pen1 = vec()
J_pen2 = vec()
rhs = scalar()
solution = scalar()
solution_n = scalar()
residual = scalar()
D_inv = scalar()
K = scalar()

n_objects = 1
n_particles = 8
elasticity = 0.3
ground_height = 0.
gravity = -9.8
penalty = 1e4
damping = 0
beta = 0.9          # for Baumgarte stabilization
dt = 0.05
learning_rate = .0002
output_directory = 'Diff_Physics'
filename='data/cuboid.obj'

@ti.layout
def place():
    ti.root.dense(ti.l,max_steps).dense(ti.i,n_objects).place(x,v,rotation,omega,v_inc,omega_inc)
    ti.root.dense(ti.i,n_objects).place(halfsize,inverse_mass,inverse_inertia)
    ti.root.dense(ti.i,n_particles).place(J_pen1,J_pen2,rhs,solution,solution_n,residual,D_inv)
    ti.root.dense(ti.l,n_particles).dense(ti.i,n_particles).place(K)
    ti.root.place(loss,friction_x,friction_z)
    ti.root.lazy_grad()

@ti.func
def rotation_matrix_x(r):
    return ti.Matrix([[1.,0.,0.],[0.,ti.cos(r),-ti.sin(r)],[0.,ti.sin(r),ti.cos(r)]])

@ti.func
def rotation_matrix_y(r):
    return ti.Matrix([[ti.cos(r),0.,ti.sin(r)],[0.,1.,0.],[-ti.sin(r),0.,ti.cos(r)]])

@ti.func
def rotation_matrix_z(r):
    return ti.Matrix([[ti.cos(r),-ti.sin(r),0.],[ti.sin(r),ti.cos(r),0.],[0.,0.,1.]])

@ti.func
def rotation_matrix(r):
    return rotation_matrix_z(r[0]) @ rotation_matrix_y(r[1]) @ rotation_matrix_x(r[2])

@ti.kernel
def initialize_properties():
    for i in range(n_objects):
        mass = 2.
        inverse_mass[i] = 1./mass
        sm = mass/12.
        inverse_inertia[i] = [[1./(sm*5),0.,0.],[0.,1./(sm*5),0.],[0.,0.,1./(sm*2)]]

@ti.func
def cross(a,b):
    return ti.Vector([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

@ti.func
def to_world(t,i,rela_x):
    rot = rotation[t, i]
    rot_matrix = rotation_matrix(rot)

    rela_pos = rot_matrix @ rela_x
    rela_v = cross(omega[t, i],rela_pos)

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos

@ti.func
def world_space_inverse_inertia(t, i):
    rot_matrix = rotation_matrix(rotation[t, i])
    return rot_matrix @ inverse_inertia[i] @ ti.transposed(rot_matrix)

@ti.func
def apply_impulse(t, i, impulse, location):
    ti.atomic_add(v_inc[t + 1, i], impulse * inverse_mass[i])
    ti.atomic_add(omega_inc[t + 1, i], world_space_inverse_inertia(t,i) @ cross(location - x[t, i], impulse))

@ti.kernel
def pseudo_integrate_state(t: ti.i32):
    for i in range(n_objects):
        v[t,i] = v[t-1,i] + dt*gravity*ti.Vector([0.,1.,0.])
        x[t,i] = x[t-1,i] + dt*v[t,i]
        omega[t,i] = omega[t-1,i]
        rotation[t,i] = rotation[t-1,i] + dt*omega[t,i]

@ti.kernel
def compute_penetration_jacobian(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in range(n_particles):
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1, k // 2 // 2 % 2 * 2 - 1])
            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)

            normal = ti.Vector([0., 1., 0.])
            rn = cross(corner_x - x[t,i], normal)
            if corner_x[1] < ground_height:
                J_pen1[k] = [0., -1., 0.]
                J_pen2[k] = [-rn[0], -rn[1], -rn[2]]
            else:
                J_pen1[k] = [0., 0., 0.]
                J_pen2[k] = [0., 0., 0.]

@ti.kernel
def compute_rhs(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in range(n_particles):
            rhs[k] = -J_pen1[k].dot(v[t,i]) - J_pen2[k].dot(omega[t,i])
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1, k // 2 // 2 % 2 * 2 - 1])
            corner_x, corner_v, rela_pos = to_world(t-1, i, offset_scale * hs)

            if(corner_x[1] < ground_height):
                rhs[k] += beta*corner_x[1]/dt

@ti.kernel
def compute_stiffness_matrix(t: ti.i32):
    for i in range(n_objects):
        for j in range(n_particles):
            for k in range(n_particles):
                ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                M_inv = ti.Matrix(ident) * inverse_mass[i]
                I_inv = world_space_inverse_inertia(t-1,i)
                K[j,k] = J_pen1[j].dot(M_inv @ J_pen1[k]) + J_pen2[j].dot(I_inv @ J_pen2[k])
            if(K[j,j] > 0.): D_inv[j] = 1./K[j,j]
            else: D_inv[j] = 0.
            solution[j] = 0.
            solution_n[j] = 0.

@ti.kernel
def solve_linear_system(t: ti.i32):
    for i in ti.static(range(100)):
        # compute residual
        for j in range(n_particles):
            residual[j] = rhs[j]
            for k in range(n_particles):
                residual[j] -= K[j,k] * solution[k]
            solution_n[j] += D_inv[j] * residual[j]
        # compute difference
        sum = 0.
        for j in range(n_particles):
            sum += (solution_n[j] - solution[j])**2
        for j in range(n_particles):
            solution[j] = solution_n[j]

@ti.kernel
def update_state(t: ti.i32):
    for i in range(n_objects):
        # compute force and torque
        force = ti.Vector([0.,0.,0.])
        torque = ti.Vector([0.,0.,0.])
        for j in ti.static(range(3)):
            for k in range(n_particles):
                force[j] += J_pen1[k][j] * solution[k]
                torque[j] += J_pen2[k][j] * solution[k]
        v[t,i] += force * inverse_mass[i]
        x[t,i] = x[t-1,i] + dt*v[t,i]
        I_inv = world_space_inverse_inertia(t-1,i)
        omega[t,i] = omega[t-1,i] + I_inv @ torque
        rotation[t,i] = rotation[t-1,i] + dt*omega[t,i]

@ti.kernel
def collide(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in ti.static(range(8)):
            fx = friction_x[None]
            fz = friction_z[None]
            # the corner for collision detection
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1, k // 2 // 2 % 2 * 2 - 1])

            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
            corner_v = corner_v + dt * gravity * ti.Vector([0., 1., 0.])

            # apply impulse so that there's no sinking
            normal = ti.Vector([0., 1., 0.])
            tao_x = ti.Vector([1., 0., 0.])
            tao_z = ti.Vector([0., 0., 1.])

            rn = cross(rela_pos, normal)
            rtx = cross(rela_pos, tao_x)
            rtz = cross(rela_pos, tao_z)
            impulse_contribution = inverse_mass[i] + rn.dot(world_space_inverse_inertia(t,i) @ rn)
            tx_impulse_contribution = inverse_mass[i] + rtx.dot(world_space_inverse_inertia(t,i) @ rtx)
            tz_impulse_contribution = inverse_mass[i] + rtz.dot(world_space_inverse_inertia(t,i) @ rtz)

            rela_v_ground = normal.dot(corner_v)

            impulse = 0.
            tx_impulse = 0.
            tz_impulse = 0.
            if rela_v_ground < 0 and corner_x[1] < ground_height:
                impulse = -(1 + elasticity) * rela_v_ground / impulse_contribution
                #if impulse > 0:
                #    tx_impulse = -corner_v.dot(tao_x) / tx_impulse_contribution
                #    tx_impulse = ti.min(fx * impulse, ti.max(-fx * impulse, tx_impulse))
                #    tz_impulse = -corner_v.dot(tao_z) / tz_impulse_contribution
                #    tz_impulse = ti.min(fz * impulse, ti.max(-fz * impulse, tz_impulse))

            # Baumgarte stabilization
            if corner_x[1] < ground_height:
                impulse = impulse - dt * penalty * (corner_x[1] - ground_height) / impulse_contribution

            apply_impulse(t,i,impulse*normal + tx_impulse*tao_x + tz_impulse*tao_z,corner_x)

@ti.kernel
def advance(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector([0., 1., 0.])
        x[t, i] = x[t - 1, i] + dt * v[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]

@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = x[t,0][0]

def write_output(frame):
    write_to_text_file(output_directory+'/info.nova-animation',frame)
    create_directory(output_directory+'/'+str(frame)+'/')
    with open(output_directory+'/'+str(frame)+'/state.txt','w') as f:
        f.write('%f %f %f\n'%(x[frame,0][0],x[frame,0][1],x[frame,0][2]))
        f.write('%f %f %f\n'%(rotation[frame,0][0],rotation[frame,0][1],rotation[frame,0][2]))

def forward():
    for t in range(1,steps):
        pseudo_integrate_state(t)
        compute_penetration_jacobian(t)
        compute_rhs(t)
        compute_stiffness_matrix(t)
        solve_linear_system(t)
        update_state(t)
        #for j in range(n_particles):
            #print('%f %f %f %f %f %f'%(J_pen1[j][0],J_pen1[j][1],J_pen1[j][2],J_pen2[j][0],J_pen2[j][1],J_pen2[j][2]))
            #print('%f'%(rhs[j]))
            #for k in range(n_particles):
            #    print(K[j,k],end=" ")
            #print()
            #print('%f'%(solution[j]))
        #print('Velocity: %f %f %f'%(v[t,0][0],v[t,0][1],v[t,0][2]))
        #print('Position: %f %f %f'%(x[t,0][0],x[t,0][1],x[t,0][2]))
        #print('Omega: %f %f %f'%(omega[t,0][0],omega[t,0][1],omega[t,0][2]))
        #print('Rotation: %f %f %f'%(rotation[t,0][0],rotation[t,0][1],rotation[t,0][2]))
        #break
        #collide(t-1)
        #advance(t)
        write_output(t)
    #loss[None] = 0
    #compute_loss(steps-1)

@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0., 0., 0.])
            omega_inc[t, i] = ti.Vector([0., 0., 0.])

def main():
    initialize_properties()
    halfsize[0] = [0.5,0.5,1.]

    friction_x[None] = 0.
    friction_z[None] = 0.
    x[0,0] = [0.,.5,0.]
    #v[0,0] = [-1.,-2.,0.]
    omega[0,0] = [0.,math.pi/3,0.]
    clear_states()

    frame=0
    create_directory(output_directory)
    create_directory(output_directory+'/common/')
    write_to_text_file(output_directory+'/common/number_of_bodies',1)
    command = 'cp '+filename+' '+output_directory+'/common/body_1.obj'
    execute_command(command)
    write_output(frame)

    forward()

if __name__ == '__main__':
    main()
