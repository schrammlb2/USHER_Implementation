import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./IT_RT.txt", sep = "\t")
df = df.fillna(555)
df = df.astype({'index': 'category', 'CSG': 'category', 'Method': 'object','Iterations': 'int32', 'Runtime': 'float32', 'Loss': 'float32'})
## Method: Nelder-Mead, L-BFGS-B

df = df[['index', 'Method', 'Iterations']]
df = df.pivot(index='index', columns='Method', values='Iterations')
ax = df.plot.bar(color = ['slateblue', 'seagreen', 'darkorange'])

# for i in [18, 22, 23]:
# # for i in [10, 14, 15]:
#     ax.get_children()[i].set_color('grey') ## white
#     ax.get_children()[i].set_edgecolor('grey')

# ## Only add one label will be sufficient
# ax.get_children()[i].set_label('Nelder-Mead exceeds limit')


for tick in ax.get_xticklabels():
    tick.set_rotation(0)

ax.set_xlabel("Trajectory")
ax.set_ylabel("Iterations")
# ax.set_ylim(0, 650)
ax.legend(frameon=False, loc=9, ncol=3, mode="expand")
plt.savefig("0-final/2-iteration.png")
# plt.show()