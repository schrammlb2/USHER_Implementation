import sys
import numpy as np
from math import pi,sin,cos
import matplotlib.pyplot as plt

motor_rpm = (330.*2.*pi)/60.
mass = 4.
gravity = 9.8
radius = .03
stall_torque = .6
dt = .01
la = .11
lb = .1
signs = np.array([1.,-1.,-1.,1.])

def desired_angular_velocity(signal):
    return (signal/255.)*motor_rpm

def compute_omega(mu,desired_velocity):
    return desired_velocity*(1. - (mu * mass * gravity * radius)/(4. * stall_torque))

def compute_matrix():
    scale = 1./radius
    return np.matrix([[-1.,1.,la+lb],[1.,1.,-(la+lb)],[-1.,1.,-(la+lb)],[1.,1.,(la+lb)]]) * scale

# compute pseudo-inverse of transformation matrix
A = np.linalg.pinv(compute_matrix())

def compute_velocity(mu,control_signals):
    # compute omega
    desired_velocity = desired_angular_velocity(control_signals*signs)
    omega = compute_omega(mu,desired_velocity)

    # compute linear and angular velocities
    x = A.dot(omega)
    return np.squeeze(np.asarray(x))

def read_control_signals(filename):
    time = []
    control_signals = []
    with open(filename,'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            values = [float(i) for i in line.split(' ')]
            time.append(np.array([values[0]]))
            control_signals.append(np.array([values[3],values[4],values[2],values[1]]))
    return time,control_signals

def read_trajectory(filename):
    pos = []
    with open(filename,'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            values = [i for i in line.split(' ')]
            if(values[1]!=''):
                pos.append(np.array([float(values[2]),float(values[3]),float(values[4])+pi]))
    return pos

if(len(sys.argv) != 3):
    print('Usage: python',sys.argv[0],'<control signal file> <real data file>')
    sys.exit(1)

time,control_signals = read_control_signals(sys.argv[1])
pos_gt = read_trajectory(sys.argv[2])

def compute_trajectory(friction):
    pos = []
    pos.append(pos_gt[0])
    dt = time[1]-time[0]
    for i in range(len(time)):
        orn = pos[i][2]
        x = compute_velocity(friction,control_signals[i])
        B = np.matrix([[cos(orn),-sin(orn),0.],[sin(orn),cos(orn),0.],[0.,0.,1.]])
        vel = B.dot(x)
        vel = np.squeeze(np.asarray(vel))
        x_pos = pos[i][0] + dt*vel[0]
        y_pos = pos[i][1] + dt*vel[1]
        orn = pos[i][2] + dt*vel[2]
        pos.append([x_pos,y_pos,orn])

    return pos

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

def main():
    poses = []
    colors = []
    symbols = []
    labels = []
    poses.append(pos_gt)
    colors.append('b')
    symbols.append('go--')
    labels.append('Ground Truth')

    mu = np.array([1.,.35,.4,.7])
    pos = compute_trajectory(mu)
    poses.append(pos)
    colors.append('r')
    symbols.append('.')
    labels.append('Prediction')
    
    draw_trajectories(poses,colors,symbols,labels)

if __name__ == '__main__':
    main()
