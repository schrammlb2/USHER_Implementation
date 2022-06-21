import pickle

import numpy as np


with open('observation.pkl', 'rb') as f:
    dat = pickle.load(f)

#print(dat[0])

# ## 3, 4, 3, 3, 4,
# obs = [robotPos, robotOrn, robotLinearVec, robotAngVEc, vec_list]

# >>> source_list = ('1','a'),('2','b'),('3','c'),('4','d')
# >>> list1, list2 = zip(*source_list)


# ## Get back the tuples
# rP, rO, rLC, rAC, vec = zip(*dat)
# print(rP)

# ## tuple to np array

# arr_rP = np.asarray(rP)
# arr_rO = np.asarray(rO)

# ## np array stack
# state = np.hstack((arr_rP, arr_rO))

# rP, rO, vec, fric = [np.asarray(x) for x in zip(*dat)]

## type: ndarray
#np.zeros(vec.shape[0], 3**4)


def format2eye(x):
    return np.hstack(np.eye(3)[x])

## ndarray to one hot...
def to_one_hot(vec):
    signed = np.sign(vec)
    replace_minus_signed = np.where(signed == -1, 2, signed)
    eyed = np.apply_along_axis(format2eye, 1, replace_minus_signed)
    return eyed


# import utils
# one_hot = utils.to_one_hot(vec)
# print(one_hot[0])
# print(one_hot.shape)

# tuple to one hot

# t  = (0,400,-400,400)
# out = utils.tuple_to_one_hot(t)
# print(out)
n = len(dat)
rP, linear_xydot, angular_wz, vec_onehot = [np.asarray(x).reshape(n,-1) for x in zip(*dat)]

# print(rP.reshape(n,-1))
# print(rO.reshape(n,-1))
# print(vec.reshape(n,-1))

# x_data =np.hstack((linear_xydot, angular_wz))
# x_data_m1 = x_data[0:n-1,:]
# print(x_data_m1.shape)
# print(x_data_m1)

# ## np difference
# rP_diff = np.diff(robotPos, axis = 0)
# print(n)
# print(robotPos[-1])
# print(rP_diff[0:2])

# ## reverse option of diff: cumsum?

# r_tmp = np.vstack((robotPos[0], rP_diff))
# r_cum = np.cumsum(r_tmp, axis = 0)
# print(r_cum[-1])

print(rP)

print(rP[0:n-1,:])
print(rP[-1])


