import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
import os


width = .8
length = 1.3
la = 1.5
lb = 2
r = .55
h = .2

def getLateralFrictionV3(f1, fd1, f2, fd2, tolerance = None):
    fd1 = np.array(fd1)
    fd2 = np.array(fd2)
    if tolerance is not None:
        fd1[np.abs(fd1) < tolerance] = 0
        fd2[np.abs(fd2) < tolerance] = 0
    f = fd1 * f1 + fd2 * f2
    if tolerance is not None:
        f[np.abs(f) < tolerance] = 0
    return f


def getWheel(linkIndexB):
    if linkIndexB >= 60:
        wheel =  "FL"
    elif linkIndexB >= 40:
        wheel =  "BL"
    elif linkIndexB >= 20:
        wheel =  "FR"
    else:
        wheel =  "BR"
    return wheel



def getPart(linkIndexB):
    if linkIndexB >= 60:
        wheel =  "FL"
    elif linkIndexB >= 40:
        wheel =  "BL"
    elif linkIndexB >= 20:
        wheel =  "FR"
    else:
        wheel =  "BR"
    pid = linkIndexB % 20
    if pid == 0:
        part =  "base"
    elif pid in [1, 3]:
        part = "plate"
    elif pid == 2:
        part = "middle_cylinder"
    elif pid in [4, 6, 8, 10, 12, 14, 16, 18]:
        part = "roller"
    elif pid in [5, 7, 9, 11, 13, 15, 17, 19]:
        part = "satelite_wheel"
    return wheel + " " + part



def GetXYTheta(pos, ori):
    """
    the position list of 3 floats [x,y,z]
    orientation as list of 4 floats in [x,y,z,w] order
    """
    r = R.from_quat(ori)
    reuler = r.as_euler('zyx', degrees=True).tolist()
    return list(pos[0:2]) + [reuler[0]]


def plotTrajectory(xyData, filename=None):
    plt.plot(xyData[...,0],xyData[...,1])
    plt.axis("equal")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def getBasename(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

# vec_dict = {0: 25, 20: 10, 40: -6, 60: 20}
def update_info(info_dict, vec_dict, fric):
    for key, value in vec_dict.items():
        info_dict['init_motorStatus'][key]['targetVelocity'] = value
        info_dict['init_physical_para'][key]['lateralFriction'] = fric
    return info_dict
        