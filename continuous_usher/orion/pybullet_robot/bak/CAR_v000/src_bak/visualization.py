import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with open('xyz_list.pkl', 'rb') as f:
    dat = pickle.load(f) ## dim k*7

with open('ori_list.pkl', 'rb') as f:
    dat2 = pickle.load(f) ## dim k*7

with open('prediction.pkl', 'rb') as f:
    prediction = pickle.load(f) ## dim k*7

dat = np.array(dat)
dat2 = np.array(dat2)
prediction = np.array(prediction[299])

print(prediction.shape)

state = np.hstack((dat,dat2))
print(state.shape)

y_predict = []
y_predict.append(state[0])
for i in range(1,len(state)-1):
    y_predict.append(state[i-1] + prediction[i])

y_predict = np.vstack(y_predict)
print(y_predict)

# xyz = y_predict[:,:3]
# x = xyz[:,0]
# y = xyz[:,1]
# z = xyz[:,2]

# ax.plot(x,y,z, 'r')

xyz = dat[:,:3]
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]
ax.scatter(x,y,z, 'b')
plt.show()

# xyz = prediction[:,:3]
# x = xyz[:,0]
# y = xyz[:,1]
# z = xyz[:,2]
# ax.plot(x,y,z, 'b')
# plt.show()