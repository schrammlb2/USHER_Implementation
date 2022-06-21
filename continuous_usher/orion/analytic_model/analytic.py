import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from math import pi,cos,sin

control_torque = 34.5575
mass = 10.
gravity = 9.8
radius = .3
stall_torque = 25.
la = .11
lb = .1
width = .152
height = .23
steps = 15

def compute_omega(mu):
    return control_torque*(1. - (mu * mass * gravity * radius)/(4. * stall_torque))

def compute_matrix():
    scale = 1./radius
    return np.matrix([[-1.,1.,la+lb],[1.,1.,-(la+lb)],[-1.,1.,-(la+lb)],[1.,1.,(la+lb)]]) * scale

def draw_rectangle(poses,colors):
    plt.cla()
    plt.axis("off")
    plt.gca().set_xlim([0,2])
    plt.gca().set_ylim([0,1.4])

    for idx,pos in enumerate(poses):
        for j in range(0,len(pos)):
            p = pos[j]
            rect = plt.Rectangle((p[0] - .5*width,p[1] - .5*height),width,height,fill=False,linewidth=1,edgecolor=colors[idx])
            t = transforms.Affine2D().rotate_around(p[0],p[1],p[2])
            rect.set_transform(t + plt.gca().transData)
            plt.gca().add_patch(rect)

            #plt.show()
            plt.savefig("./poses_{}.png".format(j))

def compute_trajectory(x_start,y_start,mu,dt):
    omega = np.zeros(4)

    for i in range(4):
        omega[i] = compute_omega(mu[i])

    A = compute_matrix()
    x = np.linalg.pinv(A).dot(omega)
    x = np.squeeze(np.asarray(x))

    pos = []
    pos.append([x_start,y_start,0.])
    for i in range(steps):
        orn = pos[i][2]
        B = np.matrix([[cos(orn),-sin(orn),0.],[sin(orn),cos(orn),0.],[0.,0.,1.]])
        vel = B.dot(x)
        vel = np.squeeze(np.asarray(vel))
        x_pos = pos[i][0] + dt*vel[0]
        y_pos = pos[i][1] + dt*vel[1]
        orn = pos[i][2] + dt*vel[2]
        pos.append([x_pos,y_pos,orn])

    return pos

def main():
    mu = np.ones(4)*0.2
    poses = []
    colors = []
    dt = .0075

    pos1 = compute_trajectory(.5*width + .01,.5*height + .01,mu,dt)
    poses.append(pos1)
    colors.append('r')

    mu[0] = 1.
    pos2 = compute_trajectory(.5,.5*height + .01,mu,dt)
    poses.append(pos2)
    colors.append('g')

    mu[3] = 1.
    pos3 = compute_trajectory(1.,.5*height + .01,mu,dt)
    poses.append(pos3)
    colors.append('b')

    mu[3] = 0.2
    mu[2] = 1.
    pos4 = compute_trajectory(1.6,.5*height + .01,mu,dt)
    poses.append(pos4)
    colors.append('m')

    draw_rectangle(poses,colors)

if __name__ == '__main__':
    main()
