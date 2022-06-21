import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import getBasename



datas = glob.glob("./dat/t*.txt")

for d in datas:
    dat = np.loadtxt(d)
    plt.plot(dat[...,0],dat[...,1], label = getBasename(d))

plt.axis("equal")
plt.legend()
plt.show()