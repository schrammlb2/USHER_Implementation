import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

def read_trajectory(filename):
    pos = []
    with open(filename,'r') as f:
        for idx,line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            values = [i for i in line.split(' ')]
            if(values[1]!=''):
                pos.append(np.array([float(values[2]),float(values[3]),float(values[4])]))
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

def draw_rectangle(poses,colors):
    xmin,xmax,ymin,ymax = compute_bounding_box(poses)
    x_length = xmax - xmin
    y_length = ymax - ymin

    plt.cla()
    plt.gca().set_xlim([xmin - .05*x_length, xmax + .05*x_length])
    plt.gca().set_ylim([ymin - .05*y_length, ymax + .05*y_length])

    width = .01*x_length
    height = .01*y_length

    for idx,pos in enumerate(poses):
        for p in pos:
            rect = plt.Rectangle((p[0] - .5*width,p[1] - .5*height),width,height,fill=False,linewidth=1,edgecolor=colors[idx])
            t = transforms.Affine2D().rotate_around(p[0],p[1],p[2])
            rect.set_transform(t + plt.gca().transData)
            plt.gca().add_patch(rect)

    plt.show()

def main():
    if(len(sys.argv) != 2):
        print('Usage: python',sys.argv[0],'<trajectory file>')
        sys.exit(1)
    filename = sys.argv[1]

    pos = read_trajectory(filename)
    poses = []
    colors = []
    poses.append(pos)
    colors.append('b')

    draw_rectangle(poses,colors)

if __name__ == '__main__':
    main()
