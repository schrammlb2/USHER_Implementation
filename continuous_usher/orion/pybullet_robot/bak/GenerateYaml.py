import yaml
import numpy as np
from pyquaternion import Quaternion
import os

def changeDynamicsFrictionValidation(paras):
    paraList = ['lateralFriction', 'spinningFriction', 'rollingFriction']
    rm_list = []
    for p in paras:
        if p not in paraList:
            print("{} is not valid friction in pybullet.changeDynamic(). REMOVED.".format(p))
            rm_list.append(p)
    for item in rm_list:
        paras.remove(item)
    return paras

def randomStartDynamics(dict_keys, paras):
    ## TODO
    # startDynamics = {'back_left': {'lateralFriction': 0.4},
    #                  'back_right': {'lateralFriction': 0.4},
    #                  'front_left': {'lateralFriction': 0.4},
    #                  'front_right': {'lateralFriction': 0.4}}
    startDynamics = {}
    paras = changeDynamicsFrictionValidation(paras)
    ##
    n = len(dict_keys)
    k = len(paras)
    for i in range(n):
        friction = (np.random.normal(0.5, 0.04,k)).round(2).tolist()
        friction = [float(_) for _ in friction]
        startDynamics[dict_keys[i]] = dict(zip(paras, friction))
    return startDynamics

def randomVelocity(dict_keys, baseVec = 400):
    action = (np.random.randint(-1,2,4) * baseVec).tolist()
    action = [int(_) for _ in action]
    out = zip(dict_keys, action)
    return dict(out)

def randomControlSignal(max_frame):
    """
    Number of Control Signal: uniform 1 to 10
    Apply to Action Frame: random from 0 to max_frame (including trimed frames)
    """
    size = np.random.randint(1,10+1)
    action_frames = np.random.randint(0, max_frame, size).tolist()
    dict_keys = ['back_left', 'back_right', 'front_left', 'front_right']
    
    cs_dict = {}
    for fr in action_frames:
        cs_dict[fr] = randomVelocity(dict_keys)
    return cs_dict

def randomStartintPos(scale):
    pos = (np.random.random_sample((3,)) - 0.5) * scale * 2 ## (from (-0.5, 0.5) -> (-scale, scale))
    ## TODO: Set z axis to a relative not high value
    ## 0.7658 is actual z axis
    pos[2] = 0.077
    return pos.tolist()
    
def randomQuarternion():
    ### CANNOT BE PURELY REANDOM!!! ( Or just throw the car back...)
    # q = Quaternion.random()
    random_degree = float(np.random.uniform(0, 360))
    q = Quaternion(axis=[0.0, 0.0, 1.0], degrees=random_degree)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
    
def dump2Yaml(dat, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(dat, f,default_flow_style=False)

def generateCSYamls(dir, mask, k, frames, robotPath, starting_i = 0):
    for i in range(starting_i, starting_i + k):
        save_path = os.path.join(dir, mask % i)
        generateTrajectoryYaml(save_path, frames, robotPath)

    print("Generated {} Control Signal Yamls in {}".format(k, dir))

def generateTrajectoryYaml(save_path, maxFrame, robotPath):
    base_dict = {'basename': 'test',
                 'robotURDF': 'simple_cuboid/simple_cuboid.URDF',
                 'sim_frames': 1000,
                 'startOrientation': [0, 0, 0, 1],
                 'startPosition': [0, 0, 0],
                 'contorl_signal': {},
                 'startDynamics': {'back_left': {'lateralFriction': 0.4},
                                   'back_right': {'lateralFriction': 0.4},
                                   'front_left': {'lateralFriction': 0.4},
                                   'front_right': {'lateralFriction': 0.4}},
                 'jointID': {0: 'back_left',
                             1: 'back_right',
                             2: 'front_right',
                             3: 'front_left'},
                 'startMotorStatus': {'back_left': {'frictionForce': 1, 'velocity': 400},
                                      'back_right': {'frictionForce': 1, 'velocity': -400},
                                      'front_left': {'frictionForce': 1, 'velocity': 400},
                                      'front_right': {'frictionForce': 1, 'velocity': -400}}
                }
    dict_keys = ['back_left', 'back_right', 'front_left', 'front_right']
    base_dict['robotURDF'] = robotPath
    base_dict['sim_frames'] = maxFrame
    base_dict['n_trim'] = 200
    base_dict['frames_before_trim'] = maxFrame + base_dict['n_trim']
    base_dict['startPosition'] = randomStartintPos(2)
    base_dict['contorl_signal'] = randomControlSignal(base_dict['frames_before_trim'])
    base_dict['startOrientation'] = randomQuarternion()
    base_dict['startDynamics'] = randomStartDynamics(dict_keys, paras=['lateralFriction'])
    if save_path is not None:
        base_dict['basename'] = os.path.splitext(os.path.basename(save_path))[0]
        dump2Yaml(base_dict, save_path)
    else:
        print(base_dict)
    
if __name__ == "__main__":
    generateTrajectoryYaml(None, 1000, '1.urdf')

   
    
    