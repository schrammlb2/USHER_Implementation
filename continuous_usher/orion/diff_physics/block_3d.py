import autograd.numpy as np
from scipy.linalg import block_diag
from autograd import grad
from scipy.optimize import minimize
from utilities import execute_command
from utilities import create_directory
from utilities import write_to_text_file

dt = 0.05
steps = 50
gravity = np.array([0,-9.8,0])
ground_normal = np.array([0.,1.,0.])
output_directory = 'Diff_Physics'

def initialize_surface(obj_filename):
    collision_proxies = []
    with open(obj_filename,'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            if(line[0]=='v'):
                line = line[2:]
                v=[float(i) for i in line.split(' ')]
                present = False
                for c in collision_proxies: 
                    if(np.linalg.norm(np.subtract(np.asarray(v),np.asarray(c)),2) < 1e-5):
                        present = True
                        break
                if(present == False): collision_proxies.append(v)

    center_of_mass = np.array([0.,0.,0.])
    for idx,proxy in enumerate(collision_proxies):
        center_of_mass += np.asarray(proxy)
    center_of_mass /= len(collision_proxies)

    return center_of_mass,collision_proxies

filename='data/cuboid.obj'
# initialize surface
center_of_mass,collision_proxies = initialize_surface(filename)

def set_mass_and_inertia():
    mass = 2.
    sm = mass/12.
    inertia_tensor = np.array([sm*5,sm*5,sm*2])
    return mass,inertia_tensor

def write_output(p,o,frame):
    write_to_text_file(output_directory+'/info.nova-animation',frame)
    create_directory(output_directory+'/'+str(frame)+'/')
    with open(output_directory+'/'+str(frame)+'/state.txt','w') as f:
        f.write('%f %f %f\n'%(p[0],p[1],p[2]))
        f.write('%f %f %f %f\n'%(o[0],o[1],o[2],o[3]))

def integrate_quaternion(orientation,angular_velocity):
    o = np.array([orientation[0],orientation[1],orientation[2],orientation[3]])
    av = np.array([angular_velocity[0],angular_velocity[1],angular_velocity[2]])

    o_vec = np.array([o[1],o[2],o[3]])
    r_vec = o[0]*av+np.cross(av,o_vec)
    o += dt*0.5*np.array([-np.dot(av,o_vec),r_vec[0],r_vec[1],r_vec[2]])
    o_norm = np.linalg.norm(o,2)
    if(o_norm > 0.): o /= o_norm
    else: o = np.array([1.,0.,0.,0.])

    return o

def rotate(q,b):
    s = q[0]
    v = np.array([q[1],q[2],q[3]])
    two_s = 2*s
    return two_s*np.cross(v,b) + (two_s*s-1.)*b + 2*np.dot(b,v)*v

def compute_colliding_proxies(position,orientation):
    world_space_colliding_proxies = []
    object_space_colliding_proxies = []
    for proxy in collision_proxies:
        point = rotate(orientation,proxy-center_of_mass)+position
        if(point[1] < 0.):
            world_space_colliding_proxies.append(point)
            object_space_colliding_proxies.append(proxy)
    return world_space_colliding_proxies,object_space_colliding_proxies

def compute_penetration_jacobian(colliding_proxies,center_of_mass):
    if(len(colliding_proxies)==0): return np.zeros((len(colliding_proxies),6))
    A = np.matrix([-ground_normal,]*len(colliding_proxies))
    B = np.cross(colliding_proxies,ground_normal) - np.cross(center_of_mass,ground_normal)
    return np.hstack((A,B))

#TODO: Compute correct normal velocity as (v + omega x r).n
#TODO: Fix Baumgarte stabilization
def compute_rhs(object_space_colliding_proxies,J,lv,av,position,orientation,linear_velocity,beta,restitution):
    vel = np.array([lv[0],lv[1],lv[2],av[0],av[1],av[2]])
    prod = np.dot(J,vel)
    rhs = -prod
    rhs += restitution*linear_velocity[1]

    #for idx,proxy in enumerate(object_space_colliding_proxies):
    #    point = rotate(orientation,object_space_colliding_proxies[idx]-center_of_mass)+position
    #    if(point[1]<0.): rhs[idx] += beta*point[1]/dt
    return rhs

def quaternion_to_matrix(q):
    s = q[0]
    v = np.array([q[1],q[2],q[3]])
    return 2*np.outer(v,v) + (2*s*s-1.)*np.eye(3,3) - 2*s*np.cross(v,np.eye(3,3))

def world_space_inertia_tensor(orientation,inertia_tensor):
    R = quaternion_to_matrix(orientation)
    RT = R.transpose()
    I = np.diag(inertia_tensor)
    return R @ I @ RT

def world_space_inverse_inertia_tensor(orientation,inverse_inertia_tensor):
    R = quaternion_to_matrix(orientation)
    RT = R.transpose()
    I_inv = np.diag(inverse_inertia_tensor)
    return R @ I_inv @ RT

def compute_inverse_mass_matrix(I_inv,mass):
    return block_diag((1./mass)*np.eye(3,3),I_inv)

def Jacobi(A,b):
    m = A.shape[0]
    n = A.shape[1]
    assert(m==n)

    # initialize x and x_new
    x = np.zeros(m)
    x_n = np.zeros(m)
    D_inv = np.diag(1./np.diag(A))

    # counter for number of iterations
    iterations = 0
    # perform Jacobi iterations until convergence
    #while True:
    for iterations in range(100):
        r = b - np.dot(A,x)
        x_n += np.dot(D_inv,r)

        # stopping criterion
        #if(np.sqrt(((x-x_n)**2).sum(dtype=np.float))<.000001): break

        # copy x_new into x
        x=x_n
        #iterations+=1

    return x

def advance_one_time_step(position,orientation,linear_velocity,angular_velocity,mass,
                          inertia_tensor,inverse_inertia_tensor,beta,restitution):
    # pseudo-integrate velocity
    lv = np.array([linear_velocity[0],linear_velocity[1],linear_velocity[2]])
    lv += dt*gravity

    # pseudo-integrate position
    p = np.array([position[0],position[1],position[2]])
    p += dt*lv

    # pseudo-integrate orientation
    av = np.array([angular_velocity[0],angular_velocity[1],angular_velocity[2]])
    o = integrate_quaternion(orientation,av)

    # compute colliding proxies
    world_space_colliding_proxies,object_space_colliding_proxies = compute_colliding_proxies(p,o)

    # compute jacobian
    J_pen = compute_penetration_jacobian(world_space_colliding_proxies,p)

    # compute right hand side
    rhs = compute_rhs(object_space_colliding_proxies,J_pen,lv,av,position,orientation,linear_velocity,beta,restitution)

    # compute inverse mass matrix
    I_inv = world_space_inverse_inertia_tensor(orientation,inverse_inertia_tensor)
    M_inv = compute_inverse_mass_matrix(I_inv,mass)

    # compute stiffness matrix
    K = J_pen @ M_inv @ J_pen.transpose()

    # solve
    x = Jacobi(K,rhs)
    F = np.dot(J_pen.transpose(),x)

    #
    S1 = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
    S2 = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

    # extract force and torque
    force = np.dot(S1,F)
    tau = np.dot(S2,F)

    # integrate velocity
    lv += force/mass

    # integrate position
    p = np.array([position[0],position[1],position[2]])
    p += dt*lv

    # integrate angular velocity
    L = np.dot(world_space_inertia_tensor(orientation,inertia_tensor),av)
    L += tau
    av = np.dot(I_inv,L)
    o = integrate_quaternion(orientation,av)

    return p,o,lv,av

def simulate(position,orientation,linear_velocity,angular_velocity,mass,
             inertia_tensor,inverse_inertia_tensor,beta,restitution):
    p = np.copy(position)
    o = np.copy(orientation)
    lv = np.copy(linear_velocity)
    av = np.copy(angular_velocity)

    for t in range(1,steps):
        p,o,lv,av = advance_one_time_step(p,o,lv,av,mass,inertia_tensor,inverse_inertia_tensor,beta,restitution)
        #write_output(p,o,t)
    return p,o,lv,av

def main():
    # set mass and inertia tensor
    mass,inertia_tensor = set_mass_and_inertia()
    inverse_inertia_tensor = 1./inertia_tensor

    #position = center_of_mass
    position = np.array([0.,5.,0.])
    linear_velocity = np.zeros(3)
    orientation = np.array([1.,0.,0.,0.])
    #angular_velocity = np.zeros(3)
    angular_velocity = np.array([0.,np.pi/3,0.])

    frame=0
    create_directory(output_directory)
    create_directory(output_directory+'/common/')
    write_to_text_file(output_directory+'/common/number_of_bodies',1)
    command = 'cp '+filename+' '+output_directory+'/common/body_1.obj'
    execute_command(command)
    write_output(position,orientation,frame)

    beta = .9                                       # bias parameter for Baumgarte stabilization
    #restitution = .9

    target_position = np.array([0.000000,0.819693,0.000000])

    def distance_from_target(p):
        return np.linalg.norm(np.subtract(p,target_position),2)

    def objective(params):
        restitution = params
        p,o,lv,av = simulate(position,orientation,linear_velocity,angular_velocity,mass,
                             inertia_tensor,inverse_inertia_tensor,beta,restitution)
        return distance_from_target(p)

    objective_with_grad = grad(objective)

    restitution = 0.
    for i in range(25):
        loss = objective(restitution)
        print('Loss: %f, Restitution: %f'%(loss,restitution))
        gradient = objective_with_grad(restitution)
        restitution -= learning_rate*gradient

    #p,o,lv,av = simulate(position,orientation,linear_velocity,angular_velocity,mass,
    #                     inertia_tensor,inverse_inertia_tensor,beta,restitution)

if __name__ == '__main__':
    main()
