
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# idx = [1, 7, 9, 11, 21]
idx = [1, 7, 9, 11, 21, 25, 27, 33]
p = re.compile(r'Final coefficients:\s+\[(.+?)\s+(.+?)\s+(.+?)\s+(.+?)\]')
pt = re.compile(r'Running time for CSG \d+:\s+(.+?)\n')
ploss = re.compile(r'Final loss:\s+\[(.+?)\]\n')
pit = re.compile(r'\s+Iterations:\s+(.+?)\n')
final_mu = []
collect_time = []
collect_loss = []
collect_iteration = []
for i in idx:
    f = open('./sigmoid/Feb24/NM_{}.out'.format(i), 'r')
    lines = f.readlines()
    for line in lines:
        m = re.search(p, line)
        mt = re.search(pt, line)
        ml = re.search(ploss, line)
        mi = re.search(pit, line)
        if m is not None:
            new = [float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4))]
            final_mu.append(new)
        if mt is not None:
            ts = float(mt.group(1))
            collect_time.append(ts)
        if ml is not None:
            loss = float(ml.group(1))
            collect_loss.append(loss)
        if mi is not None:
            ite = int(mi.group(1))
            collect_iteration.append(ite)


final_mu = np.array(final_mu)
np.savetxt("./data/df_mu.txt", final_mu)
MULIST = final_mu.tolist()
for i, mu in zip(idx, MULIST):
    print("CSG {}: np.array({})".format(i, mu))

print("runningTime: ")
for t in collect_time:
    print(t)
print("Loss: ")
for l in collect_loss:
    print(l)
print("Iterations: ")
for l in collect_iteration:
    print(l)

