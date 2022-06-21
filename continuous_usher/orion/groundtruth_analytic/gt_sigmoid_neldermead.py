import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate
import shapely.geometry as geom
import glob
import os

from GT_data_utils import loadDFfromFiles
from scipy.special import expit
import time
from datetime import datetime

motor_rpm = (330.*2.*np.pi)/60.
mass = 4.
gravity = 9.8
radius = .03
stall_torque = .6
la = .11
lb = .1
signs = np.array([1., -1., -1., 1.])
np.random.seed(7)

state_files_dir = '../camera_calibration/outputs/fixed_angle_dat_info/'
control_signal_files_dir = '../real_robot_data/Jan14_2020_Control_Signals/'

state_files_list = sorted(glob.glob(os.path.join(state_files_dir, '*.txt')))
control_signal_files_list = sorted(glob.glob(os.path.join(control_signal_files_dir, '*.csg')))
idx = list(range(len(control_signal_files_list)))

for i, rbd, csg in zip(idx, state_files_list, control_signal_files_list):
    rbd_basename = os.path.basename(rbd)
    csg_basename = os.path.basename(csg)
    # print(i, rbd_basename, csg_basename)

if (len(sys.argv) != 2):
    print("Usage: python gt_sigmoid_neldermead.py <int control_take_ith.csg>")
    sys.exit(1)
else:
    ith = int(sys.argv[1]) - 1
    state_files = [state_files_list[ith], state_files_list[ith+1]]
    control_signal_files = [control_signal_files_list[ith], control_signal_files_list[ith+1]]
    print("Evaluating ", state_files, control_signal_files)

###########################################################################

def desired_angular_velocity(signal):
    return (signal/255.)*motor_rpm

def compute_omega(mu_input, desired_velocity):
    mu = 2 * expit(mu_input)
    return desired_velocity*(1. - (mu * mass * gravity * radius)/(4. * stall_torque))


def derivative_omega(desired_velocity):
    return -(desired_velocity * mass * gravity * radius)/(4. * stall_torque)

def compute_matrix():
    scale = 1./radius
    return np.matrix([[-1., 1., la+lb], [1., 1., -(la+lb)], [-1., 1., -(la+lb)], [1., 1., (la+lb)]]) * scale

# compute pseudo-inverse of transformation matrix
A = np.linalg.pinv(compute_matrix())

def compute_velocity(mu_input, control_signals):
    # compute omega
    desired_velocity = desired_angular_velocity(control_signals*signs)
    omega = compute_omega(mu_input, desired_velocity)

    # compute linear and angular velocities
    x = A.dot(omega)
    return np.squeeze(np.asarray(x))


def read_control_signals(filename):
    timestamp = []
    control_signals = []
    with open(filename, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            values = [float(i) for i in line.split(' ')]
            timestamp.append(np.array([values[0]]))
            control_signals.append(
                np.array([values[3], values[4], values[2], values[1]]))
    return timestamp, control_signals


def read_state_info(dat):
    x = dat['pose_t_x'].to_numpy()
    y = dat['pose_t_y'].to_numpy()
    orn = dat['q_angle'].to_numpy()

    dat.drop_duplicates(subset='pose_t_x',inplace = True)
    x_new = dat['pose_t_x'].to_numpy()
    y_new = dat['pose_t_y'].to_numpy()
    smooth = 0.01
    tck, u = interpolate.splprep([x_new, y_new], s=smooth) ## default s = m-sqrt(m)
    new_sequence_length = 100
    unew = np.linspace(0, 1, new_sequence_length)
    out = interpolate.splev(unew, tck)
    a = np.transpose(np.array([out[0], out[1]]))
    line = geom.LineString(a)

    pos = []
    for i in range(len(x)):
        pos.append([x[i], y[i], (np.pi/180.)*orn[i]])
    return pos,line


print("Reading ground truth data and control signals...")
timestamp = []
control_signals = []
pos_gt = []
lines = []
for idx, csf in enumerate(control_signal_files):
    t, cs = read_control_signals(csf)
    dat_gt = loadDFfromFiles(state_files[idx])
    pos,line = read_state_info(dat_gt)
    timestamp.append(t)
    control_signals.append(cs)
    pos_gt.append(pos)
    lines.append(line)
print("Data read successfully.")


def compute_bounding_box(poses):
    xmin = 1e10
    xmax = -1e10
    ymin = 1e10
    ymax = -1e10

    for pos in poses:
        for p in pos:
            xmin = min(xmin, p[0])
            xmax = max(xmax, p[0])
            ymin = min(ymin, p[1])
            ymax = max(ymax, p[1])
    return xmin, xmax, ymin, ymax

def plt_setting(poses):
    xmin, xmax, ymin, ymax = compute_bounding_box(poses)
    x_length = abs(xmax - xmin)
    y_length = abs(ymin - ymax)
    if(x_length < .3*y_length):
        x_length = y_length

    plt.cla()
    plt.gca().set_xlim([xmin - np.sign(xmax)*.05*x_length,
                        xmin + np.sign(xmax)*(x_length + .05)])
    plt.gca().set_ylim([ymin - np.sign(ymax)*.05*y_length,
                        ymin + np.sign(ymax)*(y_length + .05)]) 

def draw_trajectories(poses, colors, symbols, labels, idx):

    for idx, pos in enumerate(poses):
        x = []
        y = []
        for p in pos:
            x.append(p[0])
            y.append(p[1])
        plt.plot(x, y, symbols[idx], color=colors[idx],
                 label=labels[idx], linewidth=2)

    plt.title("CSG {} + {}".format(int(sys.argv[1]), idx))
    # Converting datetime object to string
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%b_%d_%H_%M_%S")
    # plt.savefig("./outputs/sigmoid/{}.png".format(timestampStr))
    # plt.show()

def compute_trajectory(friction, idx):
    pos = []

    n_traj = int(len(pos_gt[idx]))
    n_sim = len(timestamp[idx])

    index_before_mapping = list(range(n_traj))
    index_after_mapping = [int(round(x/(n_traj-1)*(n_sim-1)))
                           for x in index_before_mapping]

    gt_mapping = np.zeros(n_sim)
    for i in index_after_mapping:
        gt_mapping[i] = 1.

    current_pos = pos_gt[idx][0]
    for i in range(len(timestamp[idx])):
        # compute timestamp step
        dt = timestamp[idx][1]-timestamp[idx][0]
        if(i < len(timestamp[idx])-1): dt = timestamp[idx][i+1]-timestamp[idx][i]
        else: dt = timestamp[idx][i]-timestamp[idx][i-1]

        if(gt_mapping[i] == 1.):
            pos.append(current_pos)
        orn = current_pos[2]
        x = compute_velocity(friction, control_signals[idx][i])
        B = np.matrix([[np.cos(orn), -np.sin(orn), 0.],
                       [np.sin(orn), np.cos(orn), 0.], [0., 0., 1.]])
        vel = B.dot(x)
        vel = np.squeeze(np.asarray(vel))
        x_pos = current_pos[0] + dt*vel[0]
        y_pos = current_pos[1] + dt*vel[1]
        orn = current_pos[2] + dt*vel[2]
        current_pos = [x_pos, y_pos, orn]

    return pos


def loss(friction):
    value = 0.
    for idx, p in enumerate(pos_gt):
        pos = compute_trajectory(friction, idx)
        for i in range(len(pos_gt[idx])):
            point = geom.Point(pos[i][0], pos[i][1])
            distance = lines[idx].distance(point)
            current_distance = np.sqrt((p[i][0] - pos[i][0])**2 + (p[i][1] - pos[i][1])**2)
            value += (.8*distance + .2*current_distance)
    return value


def main():
    ## Sigmoid Function: scipy.special.expit
    # np.array([0.01526036, 0.53785013, 0.93108378, 0.64608784])
    # raw_in = np.array([-4.8679844 , -1.00008319, -0.13805127, -0.73981811])
    raw_in = np.random.rand(4)
    mu0 = 2 * expit(raw_in)
    print('Initial Guess: ', mu0)
    print('Initial Loss: ', loss(raw_in))
    res = minimize(loss, raw_in, 
                   method='nelder-mead', 
                   options={'disp': True, 'maxiter': 1000, 'fatol': 1e-05})
    # res = minimize(loss, mu0, method='L-BFGS-B', jac=loss_derivative, bounds=((0.,2.),(0.,2.),(0.,2.),(0.,2.)),options={'disp': True})
    sig_output = 2 * expit(res.x)
    print('Final coefficients(raw_in): ', res.x)
    print('Final coefficients: ', sig_output)
    print('Final loss: ', loss(res.x))

    plt.cla()
    poses = []
    colors = []
    symbols = []
    labels = []

    for idx, p in enumerate(pos_gt):
        poses.append(p)
    
    plt_setting(poses)
    pose_gt1 = np.array(pos_gt[0])
    pose_gt2 = np.array(pos_gt[1])
    pred1 = np.array(compute_trajectory(res.x, 0))
    pred2 = np.array(compute_trajectory(res.x, 1))
    plt.plot(pose_gt1[...,0],pose_gt1[...,1], 'o--', markersize=2, label='Ground Truth {}'.format(int(sys.argv[1])), color = 'lightcoral')
    plt.plot(pose_gt2[...,0],pose_gt2[...,1], 'o--', markersize=2, label='Ground Truth {}'.format(int(sys.argv[1])+1), color = 'dodgerblue')
    plt.plot(pred1[...,0],pred1[...,1], '.--', color ='red',label='Prediction 1')
    plt.plot(pred2[...,0],pred2[...,1], '.--', color ='green',label='Prediction 2')

    plt.legend()
    # plt.show()
    plt.savefig("./outputs/sigmoid/CSG{}.png".format(int(sys.argv[1])))
if __name__ == '__main__':
    start = time.time()
    main()
    print("Running time for CSG {}: {}".format(sys.argv[1], time.time() - start))


    
