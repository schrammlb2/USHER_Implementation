import numpy as np
############### Global Para ###############
mass = 4.
gravity = 9.8
la = .11
lb = .1
SIGN = np.array([[1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1]])

def r_mat():
    """
    shape (4,2)
    """
    lalb = np.array([la, lb]).reshape(1,2)
    base = np.repeat(lalb, 4, axis=0)
    ## Element-wise product
    return np.multiply(base, SIGN)

def MOI_k():
    """
    float
    """
    return mass * (lb ** 2 + la ** 2) / 3.

r = r_mat()
I = MOI_k()
I_inv = 1./I

#############################################

def compute_v(State_i):
    Vx = State_i[0]
    Vy = State_i[1]
    Wz = State_i[2]
    V = np.array([Vx, Vy, 0])
    Omega = np.array([0, 0, Wz])
    v = V + np.cross(Omega, r)
    return v[:,:2]

def compute_F(v, mu):
    vx = v[:,0]
    vy = v[:,1]
    theta = np.arctan2(vy, vx) + np.pi ## (4,)
    N = mass * gravity / 4
    ## Direction is exactly inverse from V
    # tmp = np.stack((theta, np.cos(theta), np.sin(theta)), axis=1)
    # print("theta, \n", tmp)
    Fx = mu * N * np.cos(theta)
    Fy = mu * N * np.sin(theta)
    F = np.stack((Fx,Fy), axis=1)
    return F

def compute_tau(r, F):
    """
    r: np.array(4, 2)
    F: np.array(4, 2)
    """
    tau = np.cross(r, F) ## shape(4,)
    return tau

def _get_gradient(F, tau, dt):
    dV = dt * np.sum(F, axis=0) / mass ## (2,)
    dWz = dt * I_inv * np.sum(tau) ## float
    delta = np.hstack((dV, dWz))
    return delta

def visualize_v(v):
    import matplotlib.pyplot as plt
    plt.scatter(v[...,0], v[...,1])
    plt.axis('equal')
    plt.show()


def compute_gradient(mu, State_i, dt):
    """
    State_i: 
        np.array(3,)
        Vx, Vy, Wz
    """
    ## global r, dt, I <- la, lb, mass
    ## 1. Compute vi
    v = compute_v(State_i)
    # print("v", v)
    # print("sumV: ", np.sum(v, axis=0))
    # visualize_v(np.array(v))
    ## 2. Compute Fi
    F = compute_F(v, mu)
    # print("F", F)
    # print("sumF: ", np.sum(F, axis=0))
    ## 3. compute tau
    tau = compute_tau(r, F) ## shape(4,)
    # print("tau", tau)
    ## 4. compute gradient
    delta = _get_gradient(F, tau, dt)
    # print("delta", delta)
    return delta

def compute_velocity_new(mu, State_0, dt, level = 0):
    E0 = 1/2. * mass * (State_0[0]**2 + State_0[1]** 2) + 1/2. * I * State_0[2] ** 2
    # print("E0: ", E0)
    for i in range(3):
        if np.isclose(State_0[i], 0.):
            State_0[i] = 0
    if np.allclose(State_0, np.zeros(3,),rtol=1e-05, atol=1e-06):
        return np.zeros(3,)
    ## Vx, Vy, Wz
    delta = compute_gradient(mu, State_0, dt)
    State_1 = State_0 + delta
    # sign = np.sign(np.multiply(State_0, State_1))
    # num_M1 = len(np.where(sign < 0)[0])
    # if sign[0] < 0:
    #     State_1[0] = 0.
    # if sign[1] < 0:
    #     State_1[1] = 0.
    ## TODO: Wz and Vel do not comes to 0 at the same time...
    # if sign[2] < 0:
    #     State_1[2] = 0.
    for i in range(3):
        if np.isclose(State_1[i], 0.):
            State_1[i] = 0
    E1 = 1/2. * mass * (State_1[0]**2 + State_1[1]** 2) + 1/2. * I * State_1[2] ** 2
    # if level > 3:
    #     return np.zeros(3,)
    # print("E1: ", E1)
    if E1 > E0:
        print("E increased... not valid.")
        return State_0
    return State_1

if __name__ == '__main__':
    # ########## Edge Conditions ##########
    # velx = [-1, 0, 1]
    # vely = [-1, 0, 1]
    # wz = [-np.pi, -np.pi /2, 0, np.pi / 2, np.pi]
    # mu = np.array([0.21932044, 1.32255188, 1.27870286, 0.46226832])
    # # vel = [0, 0, 0]
    # for x in velx:
    #     for y in vely:
    #         for z in wz:
    #             before = [x, y, z]
    #             after = compute_velocity_new(mu, before, dt=0.01)
    #             print("{} ---> {}".format(before, after))
    #             print("Delta = {}".format(after - before))
    ##########################################################
    # mu = np.random.rand(4, )
    # mu = np.array([0.8, 0.8, .1, .1])
    mu = np.array([1.2, 1.6, 1.6, 0.9])
    # mu = np.array([0.79, 0.4, 0.74, 0.04])
    # mu = np.array([.5, .5, .5, .5])
    # mu = np.array([0.05656145, 0.06817071, 0.40644793, 0.21159967])
    # mu = np.array([0.21932044, 1.32255188, 1.27870286, 0.46226832])
    # mu = np.zeros(4,)
    # mu = np.array([.5, .5, .5, .5])
    # vel = np.random.rand(3, )
    # vel = np.array([-0.17888598168675995, -0.34150960140199615, 0.38719909456008633])
    vel = np.array([1, 1, 0])
    # vel = np.array([ 0., 1., 0])
    # vel = np.array([ 0.,0, .3])
    # vel = np.array([-0.31106346,  0.08033748,  0.32266591])
    # vel = np.array([0,0, -np.pi])
    # vel = np.array([ 0.40366715, -0.4469129,  -4.07934809])
    print("vel: ", vel)
    for i in range(30):
        vel = compute_velocity_new(mu, vel, dt=0.005)
        print("vel: ", vel)
        if np.allclose(vel, np.zeros(3,)):
            print("Decrease to 0 when in iteration {}".format(i))
            break
    print("mu", mu)
