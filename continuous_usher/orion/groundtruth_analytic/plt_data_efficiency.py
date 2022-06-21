import numpy as np
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

# CSGs=[1, 7, 9, 11, 21]
CSGs=[1, 7, 9, 11, 21, 25, 27, 33]
factor=['.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1']
index = list(range(1,9))
mapping = dict(zip(CSGs, index))


p = re.compile(r'Final Loss:\s+\[(.+?)\]\n')
headers =['index', 'CSG', 'factor', 'loss']
df = pd.DataFrame(columns=headers)

for t, i in zip(CSGs, index):
    for fac in factor:
        filename = "DE_CSG{}_f{}.out".format(t, fac)
        # print(filename)
        fp = os.path.join("./outputs/DE/", filename)

        fac100 = int(float(fac) * 100)

        f = open(fp, 'r')
        lines = f.readlines()
        for line in lines:
            m = re.search(p, line)
            if m is not None:
                loss = float(m.group(1))
                df = df.append({'index': i,
                                'CSG': t,
                                'factor': fac100,
                                'loss': loss},
                                ignore_index=True)

    df = df.astype({'index': 'category', 'CSG': 'category','factor': 'int', 'loss': 'float32'})
    # print(df)

fig, ax = plt.subplots(figsize=(12,6))
for key, grp in df.groupby(['CSG']):
    i = mapping[key]
    lab = "{}".format(i)
    ax.plot(grp['factor'], grp['loss'], 'o-', markersize=3, label=lab)
    # ax.plot(grp['iteration'], grp['loss'], label=key)

ax.set_xlabel("Percentage of Training Data", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)
ax.legend(ncol=2,frameon=False, fontsize=14)
# ax.legend(ncol=2)
# plt.show()
fig.savefig("./outputs/0-final/5-Data_Efficiency_percentage.png")