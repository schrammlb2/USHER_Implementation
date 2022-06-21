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
import time

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

# 0 2020-01-14_15-11-38.txt control_take01.csg
# 1 2020-01-14_15-14-15.txt control_take02.csg
# 2 2020-01-14_15-16-37.txt control_take03.csg
# 3 2020-01-14_15-17-17.txt control_take04.csg
# 4 2020-01-14_15-18-28.txt control_take05.csg
# 5 2020-01-14_15-19-09.txt control_take06.csg
# 6 2020-01-14_15-20-08.txt control_take07.csg
# 7 2020-01-14_15-20-39.txt control_take08.csg
# 8 2020-01-14_15-24-05.txt control_take09.csg
# 9 2020-01-14_15-25-16.txt control_take10.csg
# 10 2020-01-14_15-27-31.txt control_take11.csg
# 11 2020-01-14_15-28-09.txt control_take12.csg
# 12 2020-01-14_15-29-37.txt control_take13.csg
# 13 2020-01-14_15-30-13.txt control_take14.csg
# 14 2020-01-14_15-33-47.txt control_take15.csg
# 15 2020-01-14_15-34-22.txt control_take16.csg
# 16 2020-01-14_15-35-43.txt control_take17.csg
# 17 2020-01-14_15-36-12.txt control_take18.csg
# 18 2020-01-14_15-37-18.txt control_take19.csg
# 19 2020-01-14_15-37-49.txt control_take20.csg
# 20 2020-01-14_15-39-36.txt control_take21.csg
# 21 2020-01-14_15-40-02.txt control_take22.csg
# 22 2020-01-14_15-41-03.txt control_take23.csg
# 23 2020-01-14_15-41-31.txt control_take24.csg
# 24 2020-01-14_15-42-29.txt control_take25.csg
# 25 2020-01-14_15-42-57.txt control_take26.csg
# 26 2020-01-14_15-44-37.txt control_take27.csg
# 27 2020-01-14_15-45-14.txt control_take28.csg
# 28 2020-01-14_15-46-04.txt control_take29.csg
# 29 2020-01-14_15-46-25.txt control_take30.csg
# 30 2020-01-14_15-47-26.txt control_take31.csg
# 31 2020-01-14_15-47-54.txt control_take32.csg
# 32 2020-01-14_15-49-55.txt control_take33.csg
# 33 2020-01-14_15-50-34.txt control_take34.csg
# 34 2020-01-14_15-51-45.txt control_take35.csg
# 35 2020-01-14_15-52-04.txt control_take36.csg
if (len(sys.argv) != 2):
    print("Usage: python gt_interence_raw_fixJacobian.py <int control_take_ith.csg>")
    sys.exit(1)
else:
    ith = int(sys.argv[1]) - 1
    state_files = [state_files_list[ith], state_files_list[ith+1]]
    control_signal_files = [control_signal_files_list[ith], control_signal_files_list[ith+1]]
    print("Evaluating ", state_files, control_signal_files)

## Select one to run
## ith = 1 ~ 36
# ith = 3 - 1
# state_files = [state_files_list[ith]]
# control_signal_files = [control_signal_files_list[ith]]

### Run All
# state_files = state_files_list
# control_signal_files = control_signal_files_list

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


def compute_velocity(mu, control_signals):
    # compute omega
    desired_velocity = desired_angular_velocity(control_signals*signs)
    omega = compute_omega(mu, desired_velocity)

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
        pos.append([x[i], y[i], (pi/180.)*orn[i]])
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


def draw_trajectories(poses, colors, symbols, labels):
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
        B = np.matrix([[cos(orn), -sin(orn), 0.],
                       [sin(orn), cos(orn), 0.], [0., 0., 1.]])
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


def loss_derivative(friction):
    grad = np.zeros(4)
    for idx, p in enumerate(pos_gt):
        J = np.zeros((3, 4))

        n_traj = int(len(pos_gt[idx]))
        n_sim = len(timestamp[idx])

        index_before_mapping = list(range(n_traj))
        index_after_mapping = [int(round(x/(n_traj-1)*(n_sim-1)))
                               for x in index_before_mapping]

        gt_mapping = np.zeros(n_sim)
        for t, i in enumerate(index_after_mapping):
            gt_mapping[i] = index_before_mapping[t]

        current_pos = pos_gt[idx][0]
        for i in range(1, len(timestamp[idx])):
            # compute timestamp step
            dt = timestamp[idx][1]-timestamp[idx][0]
            if(i < len(timestamp[idx])-1): dt = timestamp[idx][i+1]-timestamp[idx][i]
            else: dt = timestamp[idx][i]-timestamp[idx][i-1]

            x = compute_velocity(friction, control_signals[idx][i])
            d_omega = derivative_omega(
                desired_angular_velocity(control_signals[idx][i]*signs))
            orn = current_pos[2]

            for j in range(4):
                J[0, j] += dt * cos(orn) * A[0, j] * d_omega[j] - dt * sin(orn) * A[1, j] * d_omega[j] - dt * (x[0] * sin(orn) + x[1] * cos(orn)) * J[2, j]
                J[1, j] += dt * sin(orn) * A[0, j] * d_omega[j] + dt * cos(orn) * A[1, j] * d_omega[j] + dt * (x[0] * cos(orn) - x[1] * sin(orn)) * J[2, j]
                ## BUG:
                # J[0,j] += dt*(cos(orn) - sin(orn))*A[0,j]*d_omega[j] - dt*(x[0]*sin(orn) + x[1]*cos(orn))*J[2,j]
                # J[1,j] += dt*(sin(orn) + cos(orn))*A[1,j]*d_omega[j] + dt*(x[0]*cos(orn) - x[1]*sin(orn))*J[2,j]
                J[2, j] += dt * A[2, j] * d_omega[j]

            if(gt_mapping[i] != 0.):
                point_gt = p[int(gt_mapping[i])]
                point = geom.Point(current_pos[0], current_pos[1])
                point_p = lines[idx].interpolate(lines[idx].project(point))
                pos_diff_gt = np.array([current_pos[0]-point_gt[0], current_pos[1]-point_gt[1]])
                pos_diff_p = np.array([current_pos[0]-point_p.x, current_pos[1]-point_p.y])
                distance_gt = max(1e-16, np.sqrt(pos_diff_gt[0]**2 + pos_diff_gt[1]**2))
                distance_p = lines[idx].distance(point)
                current_grad_gt = J[[0,1],:].transpose().dot(pos_diff_gt)
                current_grad_p = J[[0,1],:].transpose().dot(pos_diff_p)
                current_grad_gt = np.squeeze(np.asarray(current_grad_gt))
                current_grad_p = np.squeeze(np.asarray(current_grad_p))
                grad += ((.8/distance_p) * current_grad_p + (.2/distance_gt) * current_grad_gt)

            B = np.matrix([[cos(orn), -sin(orn), 0.],
                           [sin(orn), cos(orn), 0.], [0., 0., 1.]])
            vel = B.dot(x)
            vel = np.squeeze(np.asarray(vel))
            x_pos = current_pos[0] + dt*vel[0]
            y_pos = current_pos[1] + dt*vel[1]
            orn = current_pos[2] + dt*vel[2]
            current_pos = [x_pos, y_pos, orn]
    return grad


def main():
    mu0 = np.random.rand(4)
    print('Initial Guess: ', mu0)
    print('Initial Loss: ', loss(mu0))
    # res = minimize(loss, mu0, method='nelder-mead', options={'disp': True})
    res = minimize(loss, mu0, method='L-BFGS-B', jac=loss_derivative, bounds=((0.,2.),(0.,2.),(0.,2.),(0.,2.)),options={'disp': True})
    print('Final coefficients: ', res.x)
    print(loss(res.x))

    for idx, p in enumerate(pos_gt):
        poses = []
        colors = []
        symbols = []
        labels = []
        poses.append(p)
        colors.append('b')
        symbols.append('go--')
        labels.append('Ground Truth')

        pos = compute_trajectory(res.x, idx)
        poses.append(pos)
        colors.append('r')
        symbols.append('.')
        labels.append('Prediction')

        draw_trajectories(poses, colors, symbols, labels)


if __name__ == '__main__':
    start = time.time()
    main()
    print("Running time for CSG {}: {}".format(sys.argv[1], time.time() - start))



    
