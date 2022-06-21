import numpy as np
import pandas as pd 

import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt

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


filepath = "../camera_calibration/outputs/dat_info/2020-01-10_15-51-43.txt"
## 2020-01-10_15-41-02
## 2020-01-10_15-42-19
## 2020-01-10_15-43-54_360p
## 2020-01-10_15-49-16
## 2020-01-10_15-51-43 ## ???
## 2020-01-10_15-52-43

with open(filepath, 'rb') as f:
    dat_gt = pd.read_csv(f, sep=" ", header=None, names=['frames', 'pose_t_x','pose_t_y','q_angle'])

print(dat_gt)

## 1. Drop NA if any
dat = dat_gt.dropna()

## 2. Get X, Y (nd array with shape (n,))
x = dat['pose_t_x'].to_numpy()
y = dat['pose_t_y'].to_numpy()

## 3. Segment

## 4. Fit Polynomial
z = np.polyfit(x, y, 3)

print(z)

p = np.poly1d(z)

import matplotlib.pyplot as plt
xp = np.linspace(min(x)-0.1, max(x) + 0.1, 600)

# plt.plot(x, y, '.', label = 'data')
# plt.plot(xp, p(xp), '-',label='p(x)')

# plt.show()

# # ## 5. Find Closest
"""
min p(x0) - P

constraint: 1. p(x0) = y0

"""


#https://stackoverflow.com/questions/19101864/find-minimum-distance-from-point-to-complicated-curve
import scipy
from scipy.optimize import fmin_cobyla


import shapely.geometry as geom
import numpy as np

a = np.transpose(np.array([xp, p(xp)]))
print(a)

point = geom.Point(0.8, 10.5)

line = geom.LineString(a)
fig, ax = plt.subplots()
ax.plot(*a.T)
ax.axis('equal')
NearestPoint(line, ax)
plt.show()

# Note that "line.distance(point)" would be identical
## http://kitchingroup.cheme.cmu.edu/blog/2013/02/14/Find-the-minimum-distance-from-a-point-to-a-curve/
# X0 = np.array([minimum, p(minimum)])
# def objective(X):
#     x, y = X
#     return np.linalg.norm(X-point)

# def constr1(X):
#     x, y = X
#     return p(x) - y

# x0 = fmin_cobyla(objective, [0.1, p(0.1)], [constr1])
# print('The minimum distance is {0:1.2f}'.format(objective(x0)))
# print("point = ", point)
# print("x0 = ", x0)
# print("p(x0[0]) = ", p(x0[0]))
# print("constraint = ", constr1(x0))

# plt.plot(x, y, '.', label = 'data')
# plt.plot(xp, p(xp), '-',label='p(x)')
# plt.plot(point[0], point[1], 'bo', label='point')
# plt.plot(x0[0], x0[1], 'ro', label='X0')
# plt.plot([point[0], x0[0]], [point[1], x0[1]], 'b-', label='shortest distance')
# plt.axis('equal')
# # plt.xlabel('x')
# # plt.ylabel('y')
# plt.legend(loc='best')
# plt.show()

## 6. Comparing specific number of frames
# n_traj = len(x)
# n_sim = 1000

# index_before_mapping = list(range(n_traj))
# index_after_mapping = [int(round(x / (n_traj-1) * (n_sim-1))) for x in index_before_mapping]
# assert len(set(index_after_mapping)) == n_traj, "Duplicate mapping!"

# print(index_after_mapping)

# ## If simulation less than traj? Usually not but can use list(set(list)) to map

# ## https://pandas.pydata.org/pandas-docs/version/0.16/generated/pandas.DataFrame.interpolate.html
# df_inter = pd.DataFrame(index=range(n_sim), columns=['pose_t_x', 'pose_t_y', 'q_angle'])

# for i, j in zip(index_before_mapping, index_after_mapping):
#     df_inter.iloc[j] = dat_gt.iloc[i,1:4]

# df_inter = df_inter.astype('float')
# print(df_inter)
# ## ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’:
# df_tmp = df_inter.interpolate(method = 'polynomial', order=3, axis=0)
# print(df_tmp)


# xn = df_tmp['pose_t_x'].to_numpy()
# yn = df_tmp['pose_t_y'].to_numpy()

# plt.plot(xn, yn, '.-', label = 'interpolate')
# plt.plot(x, y, 'o', label = 'data')
# plt.legend(loc='best')

# plt.show()