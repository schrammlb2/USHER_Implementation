import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import sin,cos

mass = 35.7733
gravity = 9.8
radius = .03
stall_torque = 5.
dt = .01
la = .11
lb = .1
steps = 999
np.random.seed(7)

def compute_omega(mu,control_torque):
    return control_torque*(1. - (mu * mass * gravity * radius)/(4. * stall_torque))

def derivative_omega(control_torque):
    return -(control_torque * mass * gravity * radius)/(4. * stall_torque)

def compute_matrix():
    scale = 1./radius
    return np.matrix([[-1.,1.,la+lb],[1.,1.,-(la+lb)],[-1.,1.,-(la+lb)],[1.,1.,(la+lb)]]) * scale

# compute pseudo-inverse of transformation matrix
A = np.linalg.pinv(compute_matrix())

def compute_velocity(mu,control_torques):
    # compute omega
    omega = np.zeros(4)

    for i in range(4):
        omega[i] = compute_omega(mu[i],control_torques[i])

    # compute linear and angular velocities
    x = A.dot(omega)
    return np.squeeze(np.asarray(x))

def read_trajectory(filename):
    pos = []
    control_torques = []
    friction = []
    with open(filename,'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            values = [float(i) for i in line.split(' ')]
            pos.append(np.array([values[0],values[1],values[2]]))
            control_torques.append(np.array([values[6],values[7],values[8],values[9]]))
            friction = np.array([values[10],values[11],values[12],values[13]])
    return pos,control_torques,friction

def compute_bounding_box(poses):
    xmin = 1e10
    xmax = -1e10
    ymin = 1e10
    ymax = -1e10

    for pos in poses:
        for p in pos:
            xmin = min(xmin,p[0])
            xmax = max(xmax,p[0])
            ymin = min(ymin,p[1])
            ymax = max(ymax,p[1])
    return xmin,xmax,ymin,ymax

def draw_trajectories(poses,colors,symbols,labels):
    xmin,xmax,ymin,ymax = compute_bounding_box(poses)
    x_length = xmax - xmin
    y_length = ymax - ymin

    plt.cla()
    plt.gca().set_xlim([xmin - .05*x_length, xmax + .05*x_length])
    plt.gca().set_ylim([ymin - .05*y_length, ymax + .05*y_length])

    for idx,pos in enumerate(poses):
        x = []
        y = []
        for p in pos:
            x.append(p[0])
            y.append(p[1])
        plt.plot(x,y,symbols[idx],color=colors[idx],label=labels[idx],linewidth=2)

    plt.legend()
    plt.show()

if(len(sys.argv) != 2):
    print('Usage: python',sys.argv[0],'<trajectory file>')
    sys.exit(1)
filename = sys.argv[1]
pos_sim,control_torques,mu = read_trajectory(filename)
print('Ground Truth: ',mu)

def compute_trajectory(friction):
    pos = []
    pos.append(pos_sim[0])
    for i in range(steps):
        x = compute_velocity(friction,control_torques[i])
        orn = pos[i][2]
        B = np.matrix([[cos(orn),-sin(orn),0.],[sin(orn),cos(orn),0.],[0.,0.,1.]])
        vel = B.dot(x)
        vel = np.squeeze(np.asarray(vel))
        x_pos = pos[i][0] + dt*vel[0]
        y_pos = pos[i][1] + dt*vel[1]
        orn = pos[i][2] + dt*vel[2]
        pos.append([x_pos,y_pos,orn])

    return pos

pos_gt = compute_trajectory(mu)

def loss(friction):
    # check if any coefficient is negative, and penalize heavily in that case.
    is_invalid = False
    for i in range(4):
        if(friction[i] < 0.): is_invalid = True
    if(is_invalid): return np.inf

    pos = compute_trajectory(friction)
    value = 0.
    for i in range(steps+1):
        value += np.linalg.norm(np.asarray(pos[i])-np.asarray(pos_gt[i]),2)
    return value

def loss_derivative(friction):
    pos = compute_trajectory(friction)
    grad = np.zeros(4)
    J = np.zeros((3,4))
    for i in range(steps+1):
        vel = compute_velocity(friction,control_torques[i])
        pos_diff = np.asarray(pos[i])-np.asarray(pos_gt[i])
        distance = max(1e-16,np.linalg.norm(pos_diff,2))
        d_omega = derivative_omega(control_torques[i])
        orn = pos[i][2]

        for j in range(4):
            J[0,j] += dt*(cos(orn) - sin(orn))*A[0,j]*d_omega[j] - dt*(vel[0]*sin(orn) + vel[1]*cos(orn))*J[2,j]
            J[1,j] += dt*(sin(orn) + cos(orn))*A[1,j]*d_omega[j] + dt*(vel[0]*cos(orn) - vel[1]*sin(orn))*J[2,j]
            J[2,j] += dt*A[2,j]*d_omega[j]

        current_grad = J.transpose().dot(pos_diff)
        current_grad = np.squeeze(np.asarray(current_grad))
        grad += (1./distance) * current_grad
    return grad

def main():
    poses = []
    colors = []
    symbols = []
    labels = []
    poses.append(pos_gt)
    colors.append('b')
    symbols.append('go--')
    labels.append('Ground Truth')

    mu0 = np.random.rand(4)
    print('Initial Guess: ',mu0)
    print('Initial Loss: ',loss(mu0))
    #res = minimize(loss,mu0,method='nelder-mead',tol=1e-16,options={'disp': True})
    res = minimize(loss,mu0,method='BFGS',jac=loss_derivative,options={'disp': True})

    print('Final coefficients: ',res.x)
    print(loss(res.x))
    pos = compute_trajectory(res.x)
    poses.append(pos)
    colors.append('r')
    symbols.append('.')
    labels.append('Prediction')

    draw_trajectories(poses,colors,symbols,labels)

if __name__ == '__main__':
    main()
