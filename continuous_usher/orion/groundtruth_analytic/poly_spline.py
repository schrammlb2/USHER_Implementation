
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_filename_mapping import getPathofTraj
from scipy import interpolate
import shapely.geometry as geom

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
        print ('({}, {}) Distance to line: {}'.format(round(x,4), round(y,4), distance))

    def draw_segment(self, point):
        point_on_line = line.interpolate(line.project(point))
        self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y], 
                     color='red', marker='o', scalex=False, scaley=False)
        fig.canvas.draw()

### 1. Select Ground Truth data
i=1
stds, csgs = getPathofTraj(i)
# print(stds)

with open(stds, 'rb') as f:
    dat_gt = pd.read_csv(f, sep=" ", header=None, names=['frames', 'pose_t_x','pose_t_y','q_angle'])

## TODO: What if too many duplicates that related to timestep?....
dat_gt.drop_duplicates(subset='pose_t_x',inplace = True)
## 2. Get X, Y (nd array with shape (n,))
x = dat_gt['pose_t_x'].to_numpy()
y = dat_gt['pose_t_y'].to_numpy()
# Will Cause an error if x is duplicated....

## 3. Interpolate to Another length of sequence: for example, 100 here
smooth = 0.01
tck, u = interpolate.splprep([x, y], s=smooth) ## default s = m-sqrt(m)
## splprep: X --> U (0 to 1)

new_sequence_length = 100
unew = np.linspace(0, 1, new_sequence_length)

# unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)
## @ Out: list of two. out[0] is interpolated x and out[1] is interpolated y.

## 4. Interactive Find Closest
a = np.transpose(np.array([out[0], out[1]]))
line = geom.LineString(a)

## 4. [option]  Interactive Find Closest
fig, ax = plt.subplots()
ax.plot(x, y, 'b.', label='Ground Truth')
ax.plot(*a.T, 'r', label='Cubic Spline with smooth={}'.format(smooth))

ax.axis('equal')
NearestPoint(line, ax)
plt.title('CSG {} spline interpolate'.format(i))
plt.legend()
plt.show()

## 5. Numerically print distance
# point = geom.Point(1, 1)
# distance = point.distance(line)
# print ('Distance to line: {}'.format(distance))
