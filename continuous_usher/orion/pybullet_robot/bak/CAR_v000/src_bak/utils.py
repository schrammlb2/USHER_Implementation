import numpy as np
from pyquaternion import Quaternion
import torch
import pickle

def adjustQuarternion(q_pybullet):
    ## In pybullet, Q is represented as (x,y,z,w)
    ## Adjust to (w,x,y,z)
    ## tuple -> Quarternion -> tuple
    new_Q = tuple([q_pybullet[3], q_pybullet[0], q_pybullet[1], q_pybullet[2]])
    q = Quaternion(new_Q)
    return quart2tuple(q)

def quart2tuple(q):
    return tuple(np.array([q[0],q[1],q[2],q[3]]))

def frame2dict(robotPos, robotOrn, velocityList, phyParaList):
    output = {}
    output['state'] = {}
    output['state']['xyz'] = robotPos
    ## Get Orientation and tranfer from (x,y,z,w)[PyBullet] to normalized (w,x,y,z)[pyQuaternion]
    output['state']['ori'] = adjustQuarternion(robotOrn)
    output['control_signal'] = velocityList
    output['physicalPara'] = phyParaList

    return output

def dump2pickle(dat, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dat, f)

def dump2Tensor(xyz_list, ori_list, mu_list, phy_list=[], save_filename=None, saveList = None):
    k = len(xyz_list)
    assert len(ori_list) == k, "Parameter list should have same length"
    ## Generate state list
    state_list = []
    for i in range(k):
        state_list.append(xyz_list[i] + ori_list[i])
    
    ## Generate action list
    action_list = []
    if len(phy_list) != 0:
        for i in range(k):
            action_list.append(mu_list[i] + phy_list[i])
    else:
        action_list = mu_list

    ## Generate Ouput y list
    y_list = []
    for i in range(k-1):
        tmp_diff = np.array(state_list[i+1]) - np.array(state_list[i])
        y_list.append(tmp_diff.tolist())

    ## Modify series to k-1 to predict x_{t+1} - x_t
    ## IMPORTANT
    state_list = state_list[:-1]
    action_list = action_list[:-1]
    
    # Generate input list
    x_list = []
    for i in range(k-1):
        x_list.append(state_list[i] + action_list[i])

    ts_input = torch.tensor(x_list, dtype=torch.double)
    ts_output = torch.tensor(y_list, dtype=torch.double)

    if save_filename is not None:
        torch.save(ts_input, save_filename[0])
        torch.save(ts_output, save_filename[1])
    else:
        print(ts_input)
        print(ts_output)
    
    if saveList is not None:
        if len(saveList) >= 1:
            xyz_name = saveList[0]
            dump2pickle(xyz_list, xyz_name)
        if len(saveList) >= 2:
            ori_name = saveList[1]
            dump2pickle(ori_list, ori_name)
        if len(saveList) >= 3:
            mu_name = saveList[2]
            dump2pickle(mu_list, mu_name)
        if len(saveList) >= 4:
            phy_name = saveList[3]
            dump2pickle(phy_list, phy_name)

    return [ts_input, ts_output]

def format2eye(x):
    return np.hstack(np.eye(3)[x])

def to_one_hot(arr):
    ## -1, 0, 1 --> 2, 0, 1
    signed = np.sign(arr)
    replace_minus_signed = np.where(signed == -1, 2, signed)
    eyed = np.apply_along_axis(format2eye, 1, replace_minus_signed)
    return eyed

def tuple_to_one_hot(t):
    arr = np.asarray(t)
    sig = np.sign(arr)
    rep = np.where(sig == -1, 2, sig)
    eyed = np.hstack(np.eye(3)[rep])
    return eyed
