import pwlf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_filename_mapping import getPathofTraj
from scipy import interpolate

for i in range(1, 37):
# i=35
    stds, csgs = getPathofTraj(i)
    # print(stds)

    with open(stds, 'rb') as f:
        dat_gt = pd.read_csv(f, sep=" ", header=None, names=['frames', 'pose_t_x','pose_t_y','q_angle'])

    ## TODO: What if too many duplicates that related to timestep?....
    dat_gt.drop_duplicates(subset='pose_t_x',inplace = True)
    ## 2. Get X, Y (nd array with shape (n,))
    x = dat_gt['pose_t_x'].to_numpy()
    y = dat_gt['pose_t_y'].to_numpy()
    # Cause an error when x is duplicated....
    smooth = 0.01
    tck, u = interpolate.splprep([x, y], s=smooth) ## default s = m-sqrt(m)
    unew = np.arange(0, 1.01, 0.01)
    # print(unew)
    out = interpolate.splev(unew, tck)
    plt.figure()
    plt.plot(x, y, 'b', out[0], out[1], 'r')
    plt.legend(['Ground Truth', 'Cubic Spline with smooth={}'.format(smooth)])
    plt.title('CSG {} spline interpolate'.format(i))
    plt.axis('equal')
    plt.savefig('./spline_img/spline_{}.png'.format(i))