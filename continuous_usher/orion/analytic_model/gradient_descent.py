import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from scipy.optimize import minimize

control_torques = [10., -10., -10., 30.]
mass = 10.
gravity = 9.8
radius = .3
stall_torque = 25.
la = .11
lb = .1
dt = .0075
width = .152
height = .23
steps = 180
np.random.seed(7)

def compute_omega(mu,i):
    return control_torques[i]*(1. - (mu * mass * gravity * radius)/(4. * stall_torque))

def derivative_omega(i):
    return -(control_torques[i] * mass * gravity * radius)/(4. * stall_torque)

def compute_matrix():
    scale = 1./radius
    return np.matrix([[-1.,1.,la+lb],[1.,1.,-(la+lb)],[-1.,1.,-(la+lb)],[1.,1.,(la+lb)]]) * scale

# compute pseudo-inverse of transformation matrix
A = np.linalg.pinv(compute_matrix())

def compute_velocity(mu):
    # compute omega
    omega = np.zeros(4)

    for i in range(4):
        omega[i] = compute_omega(mu[i],i)

    # compute linear and angular velocities
    x = A.dot(omega)
    return np.squeeze(np.asarray(x))

# randomly choose ground truth friction coefficients
mu = np.random.rand(4)
print('Randomly selected coefficients: ',mu)
x_start = .2
y_start = .5*height + .01

def compute_trajectory(friction):
    vel = compute_velocity(friction)
    pos = []
    pos.append([x_start,y_start,0.])
    for i in range(steps):
        x_pos = pos[i][0] + dt*vel[0]
        y_pos = pos[i][1] + dt*vel[1]
        orn = pos[i][2] + dt*vel[2]
        pos.append([x_pos,y_pos,orn])

    return pos

def draw_rectangle(poses,colors):
    plt.cla()
    plt.gca().set_xlim([0,2.])
    plt.gca().set_ylim([0,2.])
    #plt.axis("off")

    for idx,pos in enumerate(poses):
        for p in pos:
            rect = plt.Rectangle((p[0] - .5*width,p[1] - .5*height),width,height,fill=False,linewidth=1,edgecolor=colors[idx])
            t = transforms.Affine2D().rotate_around(p[0],p[1],p[2])
            rect.set_transform(t + plt.gca().transData)
            plt.gca().add_patch(rect)

    plt.show()

# generate ground truth
pos_gt = compute_trajectory(mu)
vel_gt = compute_velocity(mu)
d_omega = np.zeros(4)
for i in range(4):
    d_omega[i] = derivative_omega(i)

def loss(friction):
    # check if any coefficient is negative, and penalize heavily in that case.
    is_invalid = False
    for i in range(4):
        if(friction[i] < 0.): is_invalid = True
    if(is_invalid): return np.inf

    vel = compute_velocity(friction)
    return steps * np.linalg.norm(vel-vel_gt,2)

def loss_der(friction):
    l = max(1e-16,loss(friction))
    vel = compute_velocity(friction)

    grad = A.transpose().dot(vel - vel_gt)
    grad = np.squeeze(np.asarray(grad))
    return steps * steps * (d_omega/l) * grad

def main():
    poses = []
    colors = []

    # insert ground truth trajectory
    poses.append(pos_gt)
    colors.append('b')

    mu0 = np.random.rand(4)
    print('Initial Guess: ',mu0)
    print('Initial Loss: ',loss(mu0))
    #res = minimize(loss,mu0,method='nelder-mead',tol=1e-16,options={'disp': True})
    res = minimize(loss,mu0,method='BFGS',jac=loss_der,tol=1e-16,options={'disp': True})

    print('Final coefficients: ',res.x)
    print(loss(res.x))
    prediction = compute_trajectory(res.x)
    poses.append(prediction)
    colors.append('r')

    draw_rectangle(poses,colors)

if __name__ == '__main__':
    main()
