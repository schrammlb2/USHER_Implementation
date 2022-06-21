import sys
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import pandas as pd

selected = [1, 7, 9, 11, 21, 25, 27, 33]

state_files_dir = '../camera_calibration/outputs/fixed_angle_dat_info/'
state_files_list = sorted(glob.glob(os.path.join(state_files_dir, '*.txt')))

idx = np.sort((np.array(selected)-1)).tolist()
sort_selected = np.sort((np.array(selected))).tolist()
print(sort_selected)

mapping = dict(zip(sort_selected, list(range(0, 16))))
state_files = []
for i in idx:
    state_files.append(state_files_list[i])
###########################################################################

def loadDF(filepath):
    with open(filepath, 'rb') as f:
        dat_gt = pd.read_csv(f, sep=" ", header=None, names=[
                             'x', 'y', 'angle'])
    return dat_gt


######### Get Undiff X

for i, origin in zip(sort_selected, state_files):
    plt.cla()
    f = "../NN_predictCSG/data/df/df_{}.txt".format(i)
    bn = os.path.basename(f)
    r = bn.replace('df_', '').replace('.txt', '')
    dat_gt = loadDF(f)
    cut_x = dat_gt['x'].to_numpy()
    cut_y = dat_gt['y'].to_numpy()
    plt.plot(cut_x, cut_y, 'r-.', label='Ground Truth')

    d = "./src/dat/t{0:02d}.txt".format(i)
    dat = np.loadtxt(d)
    plt.plot(dat[...,0],dat[...,1], label = "PyBullet")

    plt.legend()
    plt.axis("equal")
    pltname = "./src/img/T{}.png".format(i)
    plt.savefig(pltname)


