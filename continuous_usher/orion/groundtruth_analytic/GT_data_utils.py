import numpy as np
import pandas as pd
import shapely.geometry as geom
import matplotlib.pyplot as plt
# import scipy

#https://stackoverflow.com/questions/19101864/find-minimum-distance-from-point-to-complicated-curve
class NearestPoint(object):
    def __init__(self, line, ax):
        self.line = line
        self.ax = ax
        ax.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        x, y = event.xdata, event.ydata
        point = geom.Point(x, y)
        distance = self.line.distance(point)
        self.draw_segment(point)
        print ('Distance to line:', distance)

    def draw_segment(self, point):
        point_on_line = line.interpolate(line.project(point))
        self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y], 
                     color='red', marker='o', scalex=False, scaley=False)
        fig.canvas.draw()

def loadDFfromFiles(filepath):
    with open(filepath, 'rb') as f:
        dat_gt = pd.read_csv(f, sep=" ", header=None, names=[
                             'frames', 'pose_t_x', 'pose_t_y', 'q_angle'])

    return dat_gt


def getPolyFuncfromDat(dat, order=3, visualize=True):
    """
    return the fitted polynomial function p
    """
    # 1. Drop NA if any
    dat = dat.dropna()

    # 2. Get X, Y (nd array with shape (n,))
    x = dat['pose_t_x'].to_numpy()
    y = dat['pose_t_y'].to_numpy()

    # 3. Segment?

    # 4. Fit Polynomial
    z = np.polyfit(x, y, order)
    p = np.poly1d(z)

    if visualize:
        dat = dat_gt.dropna()

        ## 2. Get X, Y (nd array with shape (n,))
        x = dat['pose_t_x'].to_numpy()
        y = dat['pose_t_y'].to_numpy()

        xp = np.linspace(min(x), max(x), 1000)
        plt.plot(x, y, '.', label = 'data')
        plt.plot(xp, p(xp), '-',label='p(x)')

        plt.show()

    return p


def findDistance(point):
    pass


def interpolate2SimFromGT(dat_gt, n_sim, interpolate_order=3):
    """
    Map a Trajecory Data to simulation data's lentgh (By interpolate)
    Currently All Columns are using polynomial interpolate.
    Can specify different strategy from position and angle (e.g. linear) if needed
    """
    n_traj = dat_gt.shape[0]
    index_before_mapping = list(range(n_traj))
    index_after_mapping = [int(round(x / (n_traj-1) * (n_sim-1)))
                           for x in index_before_mapping]
    if n_traj <= n_sim:
        assert len(set(index_after_mapping)) == n_traj, "Duplicate mapping!"
    else:
        # If simulation less than traj? Usually not but can use list(set(list)) to map
        index_after_mapping = list(set(index_after_mapping))

    # Generate New DF
    df_inter = pd.DataFrame(index=range(n_sim), columns=[
                            'pose_t_x', 'pose_t_y', 'q_angle'])

    for i, j in zip(index_before_mapping, index_after_mapping):
        df_inter.iloc[j] = dat_gt[['pose_t_x', 'pose_t_y', 'q_angle']].iloc[i]

    df_inter = df_inter.astype('float')
    # https://pandas.pydata.org/pandas-docs/version/0.16/generated/pandas.DataFrame.interpolate.html
    # ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’:
    df_tmp = df_inter.interpolate(method='polynomial', order=interpolate_order, axis=0)
    
    return df_tmp

if __name__ == "__main__":

    ####### 1. Find Polynomial Fit #################
    filepath = "../camera_calibration/outputs/fixed_dat_info/2020-01-10_15-36-33.txt"
    dat_gt = loadDFfromFiles(filepath)
    print(dat_gt)

    p = getPolyFuncfromDat(dat_gt, 4)
    print(p)

    ####### 2-2. Map to simulation length #################
    # dat_GT_int = interpolate2SimFromGT(dat_gt, n_sim=1000, interpolate_order=3)
    # print(dat_GT_int)

    ####### 2-1. Shortest Distance #################

    x = dat_gt['pose_t_x'].to_numpy()
    y = dat_gt['pose_t_y'].to_numpy()
    # ## Shapely: Find Shortest Distance
    xp = np.linspace(min(x), max(x), 600)
    a = np.transpose(np.array([xp, p(xp)]))
    print(a)

    line = geom.LineString(a)
    point = geom.Point(0.8, 10.5)
    print (point.distance(line))

    ####### 2-3. Shortest Distance (Interactive) #################
    fig, ax = plt.subplots()
    ax.plot(*a.T)
    ax.axis('equal')
    NearestPoint(line, ax)
    plt.show()