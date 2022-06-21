
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# idx = [27, 33]
idx = [1, 7, 9, 11, 21, 25, 27, 33]
modified = list(range(1,9))
mapping = dict(zip(idx, modified))
p = re.compile(r'At iterate\s+(\d+)\s+f=\s+(.+?)\s+\|proj g\|=\s+(.+?)\n')
headers =['CSG', 'iteration', 'loss']
df = pd.DataFrame(columns=headers)

def Dexp2float(num_str):
    m = re.search(r'(.+?)D([+-]\d+)', num_str)
    base = float(m.group(1))
    scale = int(m.group(2))
    return base * 10 ** scale

for i in idx:
    f = open('./LBFGS/gt_LBFGS_CSG{}.out'.format(i), 'r')
    lines = f.readlines()
    for line in lines:
        m = re.search(p, line)
        if m is not None:
            iteration = int(m.group(1))
            f = Dexp2float(m.group(2))
            g = Dexp2float(m.group(3))
            df = df.append({'CSG': i,
                            'iteration': iteration,
                            'loss': f},
                            ignore_index=True)

df = df.astype({'CSG': 'int32','iteration': 'int32', 'loss': 'float32'})
print(df)

fig, ax = plt.subplots(figsize=(12,6))
for key, grp in df.groupby(['CSG']):
    i = mapping[key]
    lab = "{}".format(i)
    ax.plot(grp['iteration'], grp['loss'], 'o-', markersize=3, label=lab)
    # ax.plot(grp['iteration'], grp['loss'], label=key)

ax.set_xlabel("Iteration", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)
ax.legend(ncol=2,frameon=False, fontsize=14)
# ax.legend(ncol=2)
plt.show()
fig.savefig("0-final/1-LBFGS_Loss.png")