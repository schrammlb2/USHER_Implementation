import numpy as np
from pyquaternion import Quaternion

class Rigid_Body(object):
    def __init__(self,mass,friction):
        self._mass = mass
        self._friction = friction
        self._inertia_tensor = np.ones(3)
        self._inverse_inertia_tensor = np.ones(3)
        self._position = np.zeros(3)
        self._orientation = Quaternion()
        self._linear_velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)

    def initialize_surface(self,obj_filename):
        self._obj_filename = obj_filename
        self._collision_proxies = []
        with open(obj_filename,'r') as f:
            for idx,line in enumerate(f.readlines()):
                line = line.rstrip('\n')
                if(line[0]=='v'):
                    line = line[2:]
                    v=[float(i) for i in line.split(' ')]
                    vertex = np.asarray(v)
                    present = False
                    for c in self._collision_proxies:
                        if(np.linalg.norm(np.subtract(vertex,np.asarray(c)),2) < 1e-5):
                            present = True
                            break
                    if(present == False): self._collision_proxies.append(v)

        self._center_of_mass = np.array([0.,0.,0.])
        for proxy in self._collision_proxies:
            self._center_of_mass += np.asarray(proxy)
        self._center_of_mass /= len(self._collision_proxies)
        self._position=self._center_of_mass

    def compute_colliding_proxies(self,position,orientation):
        world_space_colliding_proxies = []
        object_space_colliding_proxies = []
        for proxy in self._collision_proxies:
            point = orientation.rotate(proxy-self._center_of_mass)+position
            if(point[1] < 0.):
                world_space_colliding_proxies.append(point)
                object_space_colliding_proxies.append(proxy)
        return world_space_colliding_proxies,object_space_colliding_proxies

    def compute_penetration_jacobian(self,J,colliding_proxies,normal,center_of_mass):
        for idx,proxy in enumerate(colliding_proxies):
            cross_product = np.cross(proxy-center_of_mass,normal)
            for i in range(3): J[idx][i] = -normal[i]
            for i in range(3,6): J[idx][i] = -cross_product[i-3]

    def compute_rhs(self,world_space_colliding_proxies,object_space_colliding_proxies,J,lv,av,rhs,beta,restitution,dt):
        vel = np.zeros(6)
        for i in range(3):
            vel[i]=lv[i]
            vel[i+3]=av[i]
        prod = J.dot(vel)

        for idx,proxy in enumerate(world_space_colliding_proxies):
            point = self._orientation.rotate(object_space_colliding_proxies[idx]-self._center_of_mass)+self._position
            rhs[idx] = -prod[idx]+restitution*self._linear_velocity[1]                  # assumes ground height is 0
            if(point[1]<0.): rhs[idx] += beta*point[1]/dt

    def compute_inverse_mass_matrix(self,M_inv,I_inv):
        for i in range(3): M_inv[i][i] = 1./self._mass
        for i in range(3):
            for j in range(3): M_inv[i+3][j+3] = I_inv[i][j]

    def set_position(self,position):
        self._position = position

    def set_orientation(self,orientation):
        self._orientation = orientation

    def set_linear_velocity(self,linear_velocity):
        self._linear_velocity = linear_velocity

    def set_angular_velocity(self,angular_velocity):
        self._angular_velocity = angular_velocity

    def set_inertia_tensor(self,inertia_tensor):
        self._inertia_tensor = inertia_tensor
        self._inverse_inertia_tensor = 1./inertia_tensor

    def world_space_inertia_tensor(self):
        R = self._orientation.rotation_matrix
        RT = self._orientation.conjugate.rotation_matrix
        I = np.diag(self._inertia_tensor)
        return R.dot(I.dot(RT))

    def world_space_inverse_inertia_tensor(self):
        R = self._orientation.rotation_matrix
        RT = self._orientation.conjugate.rotation_matrix
        I_inv = np.diag(self._inverse_inertia_tensor)
        return R.dot(I_inv.dot(RT))

    @property
    def mass(self):
        return self._mass

    @property
    def inertia_tensor(self):
        return self._inertia_tensor

    @property
    def inverse_inertia_tensor(self):
        return self._inverse_inertia_tensor

    @property
    def position(self):
        return self._position

    @property
    def orientation(self):
        return self._orientation

    @property
    def linear_velocity(self):
        return self._linear_velocity

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def obj_filename(self):
        return self._obj_filename
