"""
Using 5 square to represent base + 4 wheels
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
width = .8
length = 1.3
la = 1.5
lb = 2
r = .55
h = .2
# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')


# Create a Rectangle patch
base = patches.Rectangle((- width / 2., - length / 2.),width,length,linewidth=1,edgecolor='r',facecolor='none')

fl_center = (-la / 2. - h/2., lb / 2. - r / 2.)
fr_center = (la / 2.- h/2, lb / 2. - r / 2.)
bl_center = (- la / 2.- h/2, - lb / 2. - r / 2.)
br_center = (la / 2.- h/2, - lb / 2. - r / 2.)
wheels = [fl_center, fr_center, bl_center, br_center]
wheel_keys = ["FL","FR", "BL", "BR"]
wheel_centers = dict(zip(wheel_keys, wheels))

def addCar():
    ax.add_patch(base)

    # # Add the patch to the Axes
    for w in wheels:
        rec = patches.Rectangle(w, h, r,linewidth=1, edgecolor='r',facecolor='none')
        ax.add_patch(rec)

def getLine(center, value, h, r):
    v0 = center[0] + h/2.
    v1 = center[1] + r/2.
    ## Normalized
    norm = np.linalg.norm(value)
    value = value / norm
    norm = np.round(norm, 2)
    x = [v0, v0+ value[0]]
    y = [v1, v1 + value[1]]
    
    return x, y, norm

def plotFOnCar(info, linearVec):
    ax.cla()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    addCar()
    vec = np.array(linearVec[0:2])
    norm = np.linalg.norm(vec)
    vec = vec / norm
    vec = tuple(vec[0:2])
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    for key in wheel_keys:
        w = wheel_centers[key]
        f = info[key] ## np.array, (3,)
        f = f.tolist()
        f = f[0:2] ## Get first two
        x, y, norm = getLine(w, f, h, r)
        plt.plot(x, y)
        plt.text(w[0], w[1], str(norm))
        plt.pause(.2)

# Fbr = [-0.9165363253729893, 4.923011009304784]
# # Ffl = [1.3825927835724766e-05, -9.757511543609885e-06]
# Ffl = [2.0721802026325116, 13.795703083272297]
# Ffr = [0.9902692700012132, -5.228220115282315]
# Fbl = [-1.453472895300208, -8.169440295611903]
# F = [Ffl, Ffr, Fbl, Fbr]
# ## Assume the F is
# # a = -1.453472895300208
# # b = -8.169440295611903
# for w, f in zip(wheels, F):
#     x, y, norm = getLine(w, f, h, r)
#     plt.plot(x, y)
#     plt.text(w[0], w[1], str(norm))


# plt.axis("equal")

# plt.show()