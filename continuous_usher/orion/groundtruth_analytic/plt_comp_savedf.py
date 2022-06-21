import sys
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from GT_data_utils import loadDFfromFiles
from scipy import interpolate
import shapely.geometry as geom
import glob
import os
import pandas as pd

motor_rpm = (330.*2.*pi)/60.
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
    print("Usage: python gt_interence_raw_fixJacobian.py <int control_take_ith.csg>")
    sys.exit(1)
else:
    ith = int(sys.argv[1]) - 1
    state_files = [state_files_list[ith], state_files_list[ith+1]]
    control_signal_files = [control_signal_files_list[ith], control_signal_files_list[ith+1]]
    print("Evaluating ", state_files, control_signal_files)
###########################################################################
idx = [1, 7, 9, 11, 21, 25, 27, 33]
idxs = np.array(idx)
df_mu_arr = np.loadtxt("./outputs/data/df_mu.txt")
db_mu_arr = np.loadtxt("./outputs/data/db_mu.txt")
cma_mu_arr = np.loadtxt("./outputs/data/cma_mu.txt")


df_mu = df_mu_arr[np.where(idxs == ith+1)].squeeze()
db_mu = db_mu_arr[np.where(idxs == ith+1)].squeeze()
cma_mu = cma_mu_arr[np.where(idxs == ith+1)].squeeze()
print("df_mu: ", df_mu)
print("db_mu: ", db_mu)
print("cma_mu: ", cma_mu)

###########################################################################

def desired_angular_velocity(signal):
    return (signal/255.)*motor_rpm


def compute_omega(mu, desired_velocity):
    return desired_velocity*(1. - (mu * mass * gravity * radius)/(4. * stall_torque))


def derivative_omega(desired_velocity):
    return -(desired_velocity * mass * gravity * radius)/(4. * stall_torque)


def compute_matrix():
    scale = 1./radius
    return np.matrix([[-1., 1., la+lb], [1., 1., -(la+lb)], [-1., 1., -(la+lb)], [1., 1., (la+lb)]]) * scale


# compute pseudo-inverse of transformation matrix
A = np.linalg.pinv(compute_matrix())


def compute_velocity(mu, control_signals, urauns_model):
    # compute omega
    desired_velocity = desired_angular_velocity(control_signals*signs)
    if urauns_model:
        omega = desired_velocity
    else:
        omega = compute_omega(mu, desired_velocity)

    # compute linear and angular velocities
    x = A.dot(omega)
    return np.squeeze(np.asarray(x))


def read_control_signals(filename):
    time = []
    control_signals = []
    with open(filename, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.rstrip('\n')
            values = [float(i) for i in line.split(' ')]
            time.append(np.array([values[0]]))
            control_signals.append(
                np.array([values[3], values[4], values[2], values[1]]))
    return time, control_signals


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
        pos.append([x[i], y[i], (pi/180.)*orn[i]])
    return pos,line


print("Reading ground truth data and control signals...")
time = []
control_signals = []
pos_gt = []
lines = []
for idx, csf in enumerate(control_signal_files):
    t, cs = read_control_signals(csf)
    dat_gt = loadDFfromFiles(state_files[idx])
    pos,line = read_state_info(dat_gt)
    time.append(t)
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

    # plt.axis('equal')
    plt.cla()
    plt.gca().set_xlim([xmin - np.sign(xmax)*.05*x_length,
                        xmin + np.sign(xmax)*(x_length + .05)])
    plt.gca().set_ylim([ymin - np.sign(ymax)*.05*y_length,
                        ymin + np.sign(ymax)*(y_length + .05)])

def draw_trajectories(poses, colors, symbols, labels):
    xmin, xmax, ymin, ymax = compute_bounding_box(poses)
    x_length = abs(xmax - xmin)
    y_length = abs(ymin - ymax)
    if(x_length < .3*y_length):
        x_length = y_length

    # plt.axis('equal')
    # plt.cla()
    plt.gca().set_xlim([xmin - np.sign(xmax)*.05*x_length,
                        xmin + np.sign(xmax)*(x_length + .05)])
    plt.gca().set_ylim([ymin - np.sign(ymax)*.05*y_length,
                        ymin + np.sign(ymax)*(y_length + .05)])

    for idx, pos in enumerate(poses):
        x = []
        y = []
        for p in pos:
            x.append(p[0])
            y.append(p[1])
        plt.plot(x, y, symbols[idx], color=colors[idx],
                 label=labels[idx], linewidth=2)

    plt.legend()
    # plt.show()


def compute_trajectory(friction, idx, urauns_model = False):
    pos = []

    n_traj = int(len(pos_gt[idx]))
    n_sim = len(time[idx])

    index_before_mapping = list(range(n_traj))
    index_after_mapping = [int(round(x/(n_traj-1)*(n_sim-1)))
                           for x in index_before_mapping]

    gt_mapping = np.zeros(n_sim)
    for i in index_after_mapping:
        gt_mapping[i] = 1.

    current_pos = pos_gt[idx][0]
    for i in range(len(time[idx])):
        # compute time step
        dt = time[idx][1]-time[idx][0]
        if(i < len(time[idx])-1): dt = time[idx][i+1]-time[idx][i]
        else: dt = time[idx][i]-time[idx][i-1]

        if(gt_mapping[i] == 1.):
            pos.append(current_pos)
        orn = current_pos[2]
        x = compute_velocity(friction, control_signals[idx][i], urauns_model)
        B = np.matrix([[cos(orn), -sin(orn), 0.],
                       [sin(orn), cos(orn), 0.], [0., 0., 1.]])
        vel = B.dot(x)
        vel = np.squeeze(np.asarray(vel))
        x_pos = current_pos[0] + dt*vel[0]
        y_pos = current_pos[1] + dt*vel[1]
        orn = current_pos[2] + dt*vel[2]
        current_pos = [x_pos, y_pos, orn]

    return pos

def main():
    plt.cla()
    headers=['x', 'y', 'angle']

    pose_um = np.array(compute_trajectory(df_mu, idx, urauns_model = True), dtype=float)
    df_um = pd.DataFrame(pose_um, columns = headers)
    df_um['label'] = 'Uranus-Model'

    pos = compute_trajectory(df_mu, idx)
    pose_df = np.array(pos, dtype=float)
    df_pdf = pd.DataFrame(pose_df, columns = headers)
    df_pdf['label'] = 'Nelder-Mead'

    pos = compute_trajectory(db_mu, idx)
    pose_db = np.array(pos, dtype=float)
    df_pdb = pd.DataFrame(pose_db, columns = headers)
    df_pdb['label'] = 'L-BFGS-B'

    pose_cma = np.array(compute_trajectory(cma_mu, idx), dtype=float)
    df_cma = pd.DataFrame(pose_cma, columns = headers)
    df_cma['label'] = 'CMA-ES'

    # plt_setting(poses)

    pose_gt1 = np.array(pos_gt[0])
    df_gt1 = pd.DataFrame(pose_gt1, columns = headers)
    df_gt1['label'] = 'Ground Truth {}-1'.format(int(sys.argv[1]))
    pose_gt2 = np.array(pos_gt[1])
    df_gt2 = pd.DataFrame(pose_gt2, columns = headers)
    df_gt2['label'] = 'Ground Truth {}-2'.format(int(sys.argv[1])+1)

    df = pd.concat([df_cma, df_um, df_pdf, df_pdb, df_gt1, df_gt2], ignore_index=True)
    df.to_csv("./outputs/data/df_{}.csv".format(sys.argv[1]), index = False)

if __name__ == '__main__':

    main()


    