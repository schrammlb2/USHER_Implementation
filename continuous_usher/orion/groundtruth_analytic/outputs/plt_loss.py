import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./IT_RT.txt", sep = "\t")
df = df.fillna(555)
df = df.astype({'index': 'category', 'CSG': 'category', 'Method': 'object','Iterations': 'int32', 'Runtime': 'float32', 'Loss': 'float32'})
## Method: Nelder-Mead, L-BFGS-B

df = df[['index', 'Method', 'Loss']]
df = df.pivot(index='index', columns='Method', values='Loss')
ax = df.plot.bar()
# print(ax.get_children())
# childrenLS=ax.get_children()
# barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
for i in [10, 14, 15]:
    ax.get_children()[i].set_color('grey') ## white
    ax.get_children()[i].set_edgecolor('grey')

## Only add one label will be sufficient
ax.get_children()[i].set_label('Exceed Limit')
for tick in ax.get_xticklabels():
    tick.set_rotation(0)

ax.set_xlabel("Trajectory")
ax.set_ylabel("Final Loss")
ax.legend(frameon=False)
plt.savefig("0-final/5-loss.png")
# plt.show()