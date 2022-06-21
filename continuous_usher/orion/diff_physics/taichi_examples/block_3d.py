import math
import taichi as ti
from utilities import execute_command
from utilities import create_directory
from utilities import write_to_text_file

real = ti.f32
ti.set_default_fp(real)
max_steps = 4096
steps = 2
assert(steps*2 <= max_steps)

scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(3, dt=real)
quat = lambda: ti.Vector(4, dt=real)
sixlet = lambda: ti.Vector(6, dt=real)
mat = lambda: ti.Matrix(3, 3, dt=real)

loss = scalar()
x = vec()
v = vec()
rotation = quat()
omega = vec()
restitution = scalar()
mass = scalar()
inertia_tensor = mat()
inverse_inertia_tensor = mat()
center_of_mass = vec()

dt = 0.05
beta = .9                                       # bias parameter for Baumgarte stabilization
ground_normal = ti.Vector([0.,1.,0.])
gravity = ti.Vector([0,-9.8,0])
output_directory = 'Diff_Physics'
filename = '.'
n_particles = 8

collision_proxies = vec()
world_space_proxy_positions = vec()
proxy_indices = ti.global_var(ti.i32)
rhs = scalar()
J_pen = sixlet()
K = ti.global_var(ti.i32)

@ti.layout
def place():
    ti.root.dense(ti.l,max_steps).place(x,v,rotation,omega)
    ti.root.dense(ti.i,n_particles).place(collision_proxies,world_space_proxy_positions,proxy_indices,J_pen,rhs)
    ti.root.dense(ti.i,n_particles).dense(ti.i,n_particles).place(K)
    ti.root.place(loss,restitution,mass,inertia_tensor,inverse_inertia_tensor,center_of_mass)

restitution[None] = .9
mass[None] = 2.

def initialize_surface(obj_filename):
    proxies = []
    global filename
    filename = obj_filename
    with open(obj_filename,'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            if(line[0]=='v'):
                line = line[2:]
                v=[float(i) for i in line.split(' ')]
                present = False
                for c in proxies: 
                    if(math.sqrt((v[0]-c[0])**2 + (v[1]-c[1])**2 + (v[2]-c[2])**2) < 1e-5):
                        present = True
                        break
                if(present == False): proxies.append(v)

    center_of_mass[None] = [0.,0.,0.]
    for idx,proxy in enumerate(proxies):
        for i in range(3):
            center_of_mass[None][i] += proxy[i]
            collision_proxies[idx][i] = proxy[i]
    for i in range(3):
        center_of_mass[None][i] /= len(proxies)
        x[0][i] = center_of_mass[None][i]

def write_output(t: ti.i32):
    create_directory(output_directory+'/'+str(t)+'/')
    with open(output_directory+'/'+str(t)+'/state.txt','w') as f:
        f.write('%f %f %f\n'%(x[t][0],x[t][1],x[t][2]))
        f.write('%f %f %f %f\n'%(rotation[t][0],rotation[t][1],rotation[t][2],rotation[t][3]))

@ti.func
def cross(a,b):
  return ti.Vector([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

@ti.func
def quaternion_to_matrix(q):
    s = q[0]
    v = ti.Vector([q[1],q[2],q[3]])
    R = ti.Matrix(3,3)
    #return ti.Matrix([[1. - 2*v[1]**2 - 2*v[2]**2, 2*v[0]*v[1] - 2*s*v[2]    , 2*v[0]*v[2] + 2*s*v[1]],
    #                 [2*v[0]*v[1] + 2*s*v[2]    , 1. - 2*v[0]**2 - 2*v[2]**2, 2*v[1]*v[2] - 2*s*v[0]],
    #                 [2*v[0]*v[2] - 2*s*v[1]    , 2*v[1]*v[2] + 2*s*v[0]    , 1. - 2*v[0]**2 - 2*v[1]**2]])
    return R

@ti.func
def rotate(q,b):
    s = q[0]
    v = ti.Vector([q[1],q[2],q[3]])
    two_s = 2*s
    return two_s*cross(v,b) + (two_s*s-1.)*b + 2*b.dot(v)*v

@ti.func
def compute_colliding_proxies(position,orientation):
    for i in range(n_particles):
        proxy_indices[i] = 0
        object_space_point = collision_proxies[i]-center_of_mass[None]
        point = rotate(orientation,object_space_point) + position
        if(point[1] < 0.):
            proxy_indices[i]=1
            world_space_proxy_positions[i] = point

@ti.func
def zero_vec():
    return ti.Vector([0., 0., 0., 0., 0., 0.])

@ti.func
def compute_penetration_jacobian(normal,center_of_mass):
    for i in range(n_particles):
        if(proxy_indices[i]==1):
            cross_product = cross(world_space_proxy_positions[i]-center_of_mass,normal)
            J_pen[i] = [-normal[0],-normal[1],-normal[2],-cross_product[0],-cross_product[1],-cross_product[2]]
        else: J_pen[i]=zero_vec()

#TODO: Compute correct normal velocity as (v + omega x r).n
@ti.func
def compute_rhs(orientation,position,linear_velocity,lv,av):
    vel = ti.Vector([lv[0],lv[1],lv[2],av[0],av[1],av[2]])
    for i in range(n_particles):
        prod = J_pen[i].dot(vel)
        object_space_point = collision_proxies[i]-center_of_mass[None]
        point = rotate(orientation,object_space_point)+position
        rhs[i] = -prod+restitution[None]*linear_velocity[1]                 # assumes ground height is zero
        if(point[1] < 0.): rhs[i] += beta*point[1]/dt

@ti.func
def world_space_inverse_inertia_tensor(t: ti.i32):
    #rotation_matrix = quaternion_to_matrix(rotation[t-1])
    R = quaternion_to_matrix(rotation[t])
    #return rotation_matrix*inverse_inertia_tensor*ti.transposed(rotation_matrix);
    #return rotation_matrix @ inverse_inertia_tensor[None]
    #inverse_inertia_tensor[None] = rotation_matrix
    #return rotation_matrix * inverse_inertia_tensor
    #ws = R

@ti.kernel
def advance_one_time_step(t: ti.i32):
        # pseudo-integrate velocity
        lv = v[t-1]
        lv += dt*gravity

        # pseudo-integrate position
        p = x[t-1]
        p += dt*lv
        
        # pseudo-integrate orientation
        av = omega[t-1]
        avq = ti.Vector([0.,av[0],av[1],av[2]])
        o = rotation[t-1]
        o_vec = ti.Vector([o[1],o[2],o[3]])
        r_vec = avq[0]*o_vec+o[0]*av+cross(av,o_vec)
        o += dt*0.5*ti.Vector([avq[0]*o[0]-av.dot(o_vec),r_vec[0],r_vec[1],r_vec[2]])
        if(o.norm() > 0.): o /= o.norm()
        else: o = ti.Vector([1.,0.,0.,0.])

        # compute colliding proxies
        compute_colliding_proxies(p,o)

        # compute jacobian
        compute_penetration_jacobian(ground_normal,p)

        # compute right hand side
        compute_rhs(rotation[t-1],x[t-1],v[t-1],lv,av)

        # compute inverse mass matrix
        #M_inv = np.zeros((6,6))
        #world_space_inverse_inertia_tensor(t)
        #b.compute_inverse_mass_matrix(M_inv,I_inv)

        ## compute stiffness matrix
        #K = J_pen.dot(M_inv.dot(J_pen.transpose()))

        ## solve
        #x = Inverse(K).dot(rhs)
        #F = J_pen.transpose().dot(x)/dt

        ## extract force and torque
        #force = np.zeros(3)
        #tau = np.zeros(3)
        #for i in range(3):
        #    force[i]=F[i]
        #    tau[i]=F[i+3]

        ## integrate velocity
        #lv += dt*force/b.mass
        #b.set_linear_velocity(lv)

        ## integrate position
        #p = np.copy(b.position)
        #p += dt*lv
        #b.set_position(p)

        ## integrate angular velocity
        #L = b.world_space_inertia_tensor().dot(av)
        #L += dt*tau
        #av = I_inv.dot(L)
        #b.set_angular_velocity(av)

        ## integrate orientation
        #avq = Quaternion(scalar=0.,vector=av)
        #o = b.orientation
        #o += dt*0.5*avq*o
        #o = o.unit
        #b.set_orientation(o)

def forward():
    total_steps = steps
    for t in range(1,total_steps):
        advance_one_time_step(t)
        write_output(t)
        write_to_text_file(output_directory+'/info.nova-animation',t)

def main():
    initialize_surface('data/cuboid.obj')
    #print('COM:',center_of_mass[None][0],' ',center_of_mass[None][1],' ',center_of_mass[None][2])
    sm = mass[None]/12.
    inertia_tensor = [[sm*5,0.,0.],[0.,sm*5,0.],[0.,0.,sm*2]]
    inverse_inertia_tensor = [[1./(sm*5),0.,0.],[0.,1./(sm*5),0.],[0.,0.,1./(sm*2)]]
    #omega[0] = [0.,math.pi/3,0.]
    #rotation[0] = [0.92,0.,0.,0.38]

    frame=0
    create_directory(output_directory)
    create_directory(output_directory+'/common/')
    write_to_text_file(output_directory+'/info.nova-animation',frame)
    write_to_text_file(output_directory+'/common/number_of_bodies',1)
    command = 'cp '+filename+' '+output_directory+'/common/body_1.obj'
    execute_command(command)
    write_output(frame)

    forward()

if __name__ == '__main__':
    main()
