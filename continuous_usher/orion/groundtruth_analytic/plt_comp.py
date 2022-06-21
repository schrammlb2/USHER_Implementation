import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate

import pickle

CSGs=[1, 7, 9, 11, 21, 25, 27, 33]
idxs = list(range(1, 9))
mapping = dict(zip(CSGs, idxs))


def getSpline(pose_dat):
    ## 3. Interpolate to Another length of sequence: for example, 100 here
    smooth = 0.01
    tck, u = interpolate.splprep([pose_dat[...,0], pose_dat[...,1]], s=smooth) ## default s = m-sqrt(m)
    ## splprep: X --> U (0 to 1)
    unew = np.linspace(0, 1, 100)

    # unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)
    return out

def compute_bounding_box(df):
    xmin = 1e10
    xmax = -1e10
    ymin = 1e10
    ymax = -1e10

    xmin = min(xmin, df['x'].min())
    xmax = max(xmax, df['x'].max())
    ymin = min(ymin, df['y'].min())
    ymax = max(ymax, df['y'].max())
    return xmin, xmax, ymin, ymax

def plt_setting(df, idx):
    xmin, xmax, ymin, ymax = compute_bounding_box(df)
    x_length = abs(xmax - xmin)
    y_length = abs(ymin - ymax)
    if(x_length < .3*y_length):
        x_length = y_length

    
    plt.cla()
    x_lb = xmin - np.sign(xmax)*.05*x_length
    x_ub = xmin + np.sign(xmax)*(x_length + .05)
    y_lb = ymin - np.sign(ymax)*.05*y_length
    y_ub = ymin + np.sign(ymax)*(y_length + .05)
    # ## Manually adjust for CSG 33
    # if idx == 1:
    #     pass
    #     # -0.7325185391704572 0.05164462705902573 -0.10219502239695909 2.0002247297662334
    #     # x_lb = -0.75
    #     # x_ub = 0.052
    #     # y_lb = -1.1
    #     # y_ub = 2.1
    #     # print(x_lb, x_ub, y_lb, y_ub)
    # if idx == 33:
    #     print("Apply manually adjustion")
    #     x_lb = -2
    #     x_ub = 0.3
    #     y_lb = -2
    #     y_ub = 0.051
    plt.gca().set_xlim([x_lb, x_ub])
    plt.gca().set_ylim([y_lb, y_ub])
    # plt.axis('equal')
    # plt.gca().set_aspect('equal', adjustable='datalim')

def main(idx):
    """
    idx in [1, 7, 9, 11, 21, 25, 27, 33]
    newID is [1, 2, 3, 4, ..., 8] (Used to plot)
    """
    newID = mapping[idx]
    df = pd.read_csv("./outputs/data/df_{}.csv".format(idx))
    # plt_setting(df, idx)
    plt.cla()
    ################ Experiment data ###########################
    pose_gt1 = df[df['label'].str.endswith("-1")].filter(['x', 'y', 'angle']).to_numpy(dtype=float)
    pose_gt2 = df[df['label'].str.endswith("-2")].filter(['x', 'y', 'angle']).to_numpy(dtype=float)
    pose_db = df[df['label'].str.startswith("L")].filter(['x', 'y', 'angle']).to_numpy(dtype=float)
    pose_df = df[df['label'].str.startswith("N")].filter(['x', 'y', 'angle']).to_numpy(dtype=float)
    pose_um = df[df['label'].str.startswith("U")].filter(['x', 'y', 'angle']).to_numpy(dtype=float)
    pose_cma = df[df['label'].str.startswith("C")].filter(['x', 'y', 'angle']).to_numpy(dtype=float)

    sigma = .1
    sigma2 = sigma * 2
    sigma3 = sigma * 3

    #################### Plotting ##########################

    plt.plot(pose_gt1[...,0],pose_gt1[...,1], '-.', markersize=1, 
             label='Ground Truth {}-{}'.format(newID, 1), color = 'lightcoral')
    plt.plot(pose_gt2[...,0],pose_gt2[...,1], '-.', markersize=1, 
             label='Ground Truth {}-{}'.format(newID, 2), color = 'dodgerblue')

    plt.plot(pose_db[...,0], pose_db[...,1], 'o', markersize=2, label='Our Method', color = 'green')
    # plt.fill_between(pose_db[...,0],pose_db[...,1] - sigma, pose_db[...,1]+sigma, facecolor='green', alpha=0.2)

    # plt.plot(pose_df[...,0],pose_df[...,1], 'o', markersize=2, label='Nelder-Mead', color = 'orange')
    # plt.fill_between(pose_df[...,0],pose_df[...,1] - sigma2, pose_df[...,1]+sigma2, facecolor='orange', alpha=0.2)
    
    plt.plot(pose_um[...,0], pose_um[...,1], 'o', markersize=2, label='Uranus-Model', color = 'violet')
    # plt.fill_between(pose_um[...,0],pose_um[...,1] - sigma, pose_um[...,1]+sigma, facecolor='darkviolet', alpha=0.2)

    # plt.plot(pose_cma[...,0], pose_cma[...,1], 'o', markersize=2, label='CMA-ES', color = 'royalblue')
    # plt.fill_between(pose_cma[...,0],pose_cma[...,1] - sigma2, pose_cma[...,1]+sigma2, facecolor='royalblue', alpha=0.2)


    ################ NN data from 20201118 ##############################
    pospath = '20201118_NNdata/traj/{}_1pos'.format(newID)
    angpath = '20201118_NNdata/traj/{}_1ang'.format(newID)
    with open(pospath, 'rb') as pickle_file:
        poses = pickle.load(pickle_file)
    with open(angpath, 'rb') as pickle_file:
        loads = pickle.load(pickle_file)

    plt.plot(poses[:, 0], poses[:, 1], 'o', markersize=2, color='red', label='NN Prediction')

    ######################################################################

    # if idx == 1:
    #     plt.legend(frameon=False, fontsize=11, loc='lower left')
    # else:
    #     plt.legend(frameon=False, fontsize=11)

    plt.legend(frameon=False, fontsize=11)

    # plt.legend(frameon=False, fontsize=11, ncol=3, mode="expand")
    plt.axis("equal")
    # plt.show()
    plt.savefig("./outputs/V20201118/comp_CSG{}.png".format(idx))
    

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        # print("Usage: python gt_interence_raw_fixJacobian.py <int control_take_ith.csg>")
        # sys.exit(1)
        for i in CSGs:
            main(i)
    else:
        idx = int(sys.argv[1])
        ith = idx - 1
        newID = mapping[idx]

        main(idx)

    

    