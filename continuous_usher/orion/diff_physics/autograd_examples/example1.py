import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize

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

def set_mass_and_inertia():
    mass = 2.
    sm = mass/12.
    inertia_tensor = np.array([sm*5,sm*5,sm*2])
    return mass,inertia_tensor

def advance_one_time_step(position1,velocity1,gravity1,position2,velocity2,gravity2,dt):
    velocity1 += gravity1*dt
    position1 += velocity1*dt
    velocity2 += gravity2*dt
    position2 += velocity2*dt
    return position1,velocity1,position2,velocity2

def simulate(position1,velocity1,gravity1,position2,velocity2,gravity2,dt,steps):
    p1 = np.copy(position1)
    v1 = np.copy(velocity1)
    p2 = np.copy(position2)
    v2 = np.copy(velocity2)
    for i in range(steps):
        p1,v1,p2,v2 = advance_one_time_step(p1,v1,gravity1,p2,v2,gravity2,dt)
    return p1,p2

def main():
    filename='data/cuboid.obj'
    # initialize surface
    center_of_mass,collision_proxies = initialize_surface(filename)

    # set mass and inertia tensor
    mass,inertia_tensor = set_mass_and_inertia()
    inverse_inertia_tensor = 1./inertia_tensor

    position1 = np.array([0.,5.,0.])
    position2 = np.array([0.,5.,0.])
    velocity1 = np.zeros(3)
    velocity2 = np.zeros(3)
    orientation = np.array([1.,0.,0.,0.])
    target_position1 = np.array([0.,0.,0.])
    target_position2 = np.array([0.,0.,0.])
    dt = 0.05

    def distance_from_target(p1,p2):
        #return np.linalg.norm(np.subtract(p,target_position),2)
        return (p1[1]-target_position1[1])+(p2[1]-target_position2[1])

    def objective(params):
        gravity1 = np.array([params[0],params[1],params[2]])
        gravity2 = np.array([params[3],params[4],params[5]])
        target_pos1,target_pos2 = simulate(position1,velocity1,gravity1,position2,velocity2,gravity2,dt,50)
        return distance_from_target(target_pos1,target_pos2)

    objective_with_grad = grad(objective)

    #print("Optimizing...")
    initial_gravity = np.zeros(6)
    #result = minimize(objective_with_grad,initial_gravity,jac=True, method='CG',options={'maxiter':25, 'disp':True})

    print(objective_with_grad(initial_gravity))

if __name__ == '__main__':
    main()
