import numpy as np
import taichi as ti
import rigid_body as rb
from pyquaternion import Quaternion
from utilities import execute_command
from utilities import create_directory
from utilities import write_to_text_file

real = ti.f32
ti.set_default_fp(real)
max_steps = 4096
steps = 100
assert(steps*2 <= max_steps)

dt = 0.05
beta = .9                                       # bias parameter for Baumgarte stabilization
restitution = .9
ground_normal = np.array([0.,1.,0.])
gravity = np.array([0,-9.8,0])
output_directory = 'Diff_Physics'

bodies = []
bodies.append(rb.Rigid_Body(2.,1.))
bodies[0].initialize_surface('data/cuboid.obj')
bodies[0].set_position(np.array([0.,5.,0.]))
bodies[0].set_orientation(Quaternion(axis=[0,0,1],angle=3.14159265/4))
bodies[0].set_linear_velocity(np.array([2.,10.,0.]))
bodies[0].set_angular_velocity(np.array([0.,np.pi/3,0.]))
sm = bodies[0].mass/12.
bodies[0].set_inertia_tensor(np.array([sm*5,sm*5,sm*2]))

def write_output(frame):
    create_directory(output_directory+'/'+str(frame)+'/')
    for b in bodies:
        p = b.position
        o = b.orientation
        with open(output_directory+'/'+str(frame)+'/state.txt','w') as f:
            f.write('%f %f %f\n'%(p[0],p[1],p[2]))
            f.write('%f %f %f %f\n'%(o[0],o[1],o[2],o[3]))

def Gauss_Seidel(A,b):
    m = A.shape[0]
    n = A.shape[1]
    assert(m==n)

    # initialize x and x_new
    x = np.zeros(m)
    x_n = np.zeros(m)

    # counter for number of iterations
    iterations = 0
    # perform Gauss-Seidel iterations until convergence
    while True:
        for i in range(0,m):
            x_n[i] = b[i]/A[i,i]
            sum = 0
            for j in range(0,m):
                if(j<i): sum+=A[i,j]*x_n[j]
                if(j>i): sum+=A[i,j]*x[j]
            x_n[i] -=sum/A[i,i]

        # stopping criterion
        if(np.linalg.norm(x-x_n,2)<.000001): break

        # copy x_new into x
        for i in range(0,m): x[i]=x_n[i]
        iterations+=1

    return x

def advance_one_time_step(t: ti.i32):
    for b in bodies:
        # pseudo-integrate velocity
        lv = np.copy(b.linear_velocity)
        lv += dt*gravity

        # pseudo-integrate position
        p = np.copy(b.position)
        p += dt*lv
        
        # pseudo-integrate orientation
        av = np.copy(b.angular_velocity)
        avq = Quaternion(scalar=0.,vector=av)
        o = b.orientation
        o += dt*0.5*avq*o
        o = o.unit

        # compute colliding proxies
        world_space_colliding_proxies,object_space_colliding_proxies = b.compute_colliding_proxies(p,o)

        # compute jacobian
        J_pen = np.zeros((len(world_space_colliding_proxies),6))
        b.compute_penetration_jacobian(J_pen,world_space_colliding_proxies,ground_normal,p)

        # compute right hand side
        rhs = np.zeros(len(world_space_colliding_proxies))
        b.compute_rhs(world_space_colliding_proxies,object_space_colliding_proxies,J_pen,lv,av,rhs,beta,restitution,dt)

        # compute inverse mass matrix
        M_inv = np.zeros((6,6))
        I_inv = b.world_space_inverse_inertia_tensor()
        b.compute_inverse_mass_matrix(M_inv,I_inv)

        # compute stiffness matrix
        K = J_pen.dot(M_inv.dot(J_pen.transpose()))

        # solve
        x = Gauss_Seidel(K,rhs)
        F = J_pen.transpose().dot(x)/dt

        # extract force and torque
        force = np.zeros(3)
        tau = np.zeros(3)
        for i in range(3):
            force[i]=F[i]
            tau[i]=F[i+3]

        # integrate velocity
        lv += dt*force/b.mass
        b.set_linear_velocity(lv)

        # integrate position
        p = np.copy(b.position)
        p += dt*lv
        b.set_position(p)

        # integrate angular velocity
        L = b.world_space_inertia_tensor().dot(av)
        L += dt*tau
        av = I_inv.dot(L)
        b.set_angular_velocity(av)

        # integrate orientation
        avq = Quaternion(scalar=0.,vector=av)
        o = b.orientation
        o += dt*0.5*avq*o
        o = o.unit
        b.set_orientation(o)

def forward():
    total_steps = steps
    for t in range(1,total_steps):
        advance_one_time_step(t)
        write_output(t)
        write_to_text_file(output_directory+'/info.nova-animation',t)

def main():
    frame=0
    create_directory(output_directory)
    create_directory(output_directory+'/common/')
    write_to_text_file(output_directory+'/info.nova-animation',frame)
    write_to_text_file(output_directory+'/common/number_of_bodies',len(bodies))
    for i,body in enumerate(bodies):
        command = 'cp '+body.obj_filename+' '+output_directory+'/common/body_'+str(i+1)+'.obj'
        execute_command(command)
    write_output(frame)

    forward()

if __name__ == '__main__':
    main()
