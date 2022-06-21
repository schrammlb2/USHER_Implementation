
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# idx = [27, 33]
idx = [1, 7, 9, 11, 21, 25, 27, 33]
# finalF = re.compile(r'\s+F\s+=\s+(.+?)\n')
# runningTime = re.compile(r'Running time for CSG \d+:\s+(.+?)\n')
# info = re.compile(r'\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+?)\s+(.+?)\n')
p = re.compile(r'Final coefficients:\s+\[(.+?)\s+(.+?)\s+(.+?)\s+(.+?)\]')
headers =['CSG', 'iteration', 'loss']
df = pd.DataFrame(columns=headers)

final_mu = []
for i in idx:
    f = open('./LBFGS/gt_LBFGS_CSG{}.out'.format(i), 'r')
    lines = f.readlines()
    for line in lines:
        m = re.search(p, line)
        if m is not None:
            # Tit = int(m.group(2))
            # Tnf = int(m.group(3))
            # print(Tit)
            new = [float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4))]
            final_mu.append(new)

final_mu = np.array(final_mu)
print(final_mu.shape)
np.savetxt("./data/db_mu.txt", final_mu)
MULIST = final_mu.tolist()
for i, mu in zip(idx, MULIST):
    print("CSG {}: np.array({})".format(i, mu))
