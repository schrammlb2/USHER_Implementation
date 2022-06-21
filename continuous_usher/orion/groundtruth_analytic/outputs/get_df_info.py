
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

idx = [1, 7, 9, 11, 21, 25, 27, 33]
# get_iter = re.compile(r'\s+Iterations:\s+(\d+)')
p = re.compile(r'^\[(.+?)\]\n')
# runningTime = re.compile(r'Running time for CSG \d+:\s+(.+?)\n')
headers =['CSG', 'iteration', 'loss']
df = pd.DataFrame(columns=headers)

f = open('./sigmoid/report.out','r')
lines = f.readlines()
final_mu = []
for line in lines:
    m = re.search(p, line)
    if m is not None:
        fl = float(m.group(1))
        print(fl)
        # new = [float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4))]
        # final_mu.append(new)


# final_mu = np.array(final_mu)
# print(final_mu.shape)
# print(final_mu)