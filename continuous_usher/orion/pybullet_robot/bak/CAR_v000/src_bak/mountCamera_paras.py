import numpy as np
import pybullet as p
import pybullet_data
import time

import torch

import pickle
from pyquaternion import Quaternion
import utils

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Add plane
plane_id = p.loadURDF("plane.urdf")

# Add kuka bot
start_pos = [0, 0, 0.1]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
#robotId = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)
robotId = p.loadURDF("CAR_v000.urdf", start_pos, start_orientation)

fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

## Set Parameter lists
# #### SetJointMotor
  # ## bot_motor:51; neck_motor: 53; back_l: 35; back_R:2; front_L: 24;front_R:13

gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []
maxForce = 500
wheelIds_dict={}
controlTargetIds_dict={}
wheelIds_dict["back_left"] = dict()
wheelIds_dict["back_right"] = dict()
wheelIds_dict["front_left"] = dict()
wheelIds_dict["front_right"] = dict()
controlTargetIds_dict["Neck_Motor"] = dict()
controlTargetIds_dict["Bot_Motor"] = dict()

default_v = 400

wheelIds_dict["back_left"]["id"] = 35   ## ->
wheelIds_dict["back_right"]["id"] = 2   ## -< 
wheelIds_dict["front_left"]["id"] = 24  ## ->
wheelIds_dict["front_right"]["id"] = 13 ## -<
wheelIds_dict["back_left"]["default_velocity"] = default_v   ## ->
wheelIds_dict["back_right"]["default_velocity"] = -default_v   ## -< 
wheelIds_dict["front_left"]["default_velocity"] = default_v  ## ->
wheelIds_dict["front_right"]["default_velocity"] = -default_v ## -<
controlTargetIds_dict["Neck_Motor"]["id"] = 53
controlTargetIds_dict["Bot_Motor"]["id"] = 51

for key, wheel in wheelIds_dict.items():
    wheel["vecParaId"] = p.addUserDebugParameter(key + "_veclocity", -500, 500, wheel["default_velocity"])

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(robotId, -1, linearDamping=0, angularDamping=0)


for j in range(p.getNumJoints(robotId)):
    p.changeDynamics(robotId, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(robotId, j)

for key, controlTarget in controlTargetIds_dict.items():
  controlTarget["positionID"] = p.addUserDebugParameter(key + "_position", -4, 4, 0)

p.setRealTimeSimulation(1)

#p.setGravity(0, 0, -10)
p.setTimeStep(0.01)

def kuka_camera():
    # Center of mass position and orientation (of link-7)
    com_p, com_o, _, _, _, _ = p.getLinkState(robotId, 55,computeForwardKinematics=True)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (1, 0, 0) # x-axis
    init_up_vector = (0, 0, 1) # z-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p + 1 * camera_vector, com_p + 5 * camera_vector, up_vector)
    img = p.getCameraImage(240, 240, view_matrix, projection_matrix)
    return img


info_dict = {}

xyz_list = []
ori_list = []
mu_list = []
phy_list = []

# Main loop
for t in range(1000):
    p.stepSimulation()
    #kuka_camera()
    p.setGravity(0, 0, p.readUserDebugParameter(gravId))
    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        #p.setJointMotorControl2(robotId, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
    
    for key, wheel in wheelIds_dict.items():
        vecParaId=wheel["vecParaId"]
        targetVec = p.readUserDebugParameter(vecParaId)
        p.setJointMotorControl2(bodyUniqueId=robotId, jointIndex=wheel["id"], controlMode=p.VELOCITY_CONTROL, targetVelocity = targetVec, force = maxForce)

    for key, controlTarget in controlTargetIds_dict.items():
        c = controlTarget["positionID"]
        targetPos = p.readUserDebugParameter(c)
        p.setJointMotorControl2(robotId, controlTarget["id"], p.POSITION_CONTROL, targetPos, force=5 * 240.)

    ######### Get serial data
    robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
    robotLinearVec, robotAngVEc = p.getBaseVelocity(robotId) ## [[xdot,ydot,zdot], [wx,wy,wz]]

    ############# For dict --> Draw in Blender
    vec_list = []
    friction_list = []
    for key, wheel in wheelIds_dict.items():
        ## Control Signal MU
        targetVec = p.readUserDebugParameter(wheel["vecParaId"])
        vec_list.append(targetVec)
        ## Physical Parameter Theta
        jointInfoList = p.getJointInfo(robotId, wheel["id"])
        friction_list.append(jointInfoList[7])

    info_dict[t] = utils.frame2dict(robotPos, robotOrn, vec_list, friction_list)
    #############################
    ### Generate list for further use
    xyz_list.append(list(robotPos))
    ori_list.append(list(utils.adjustQuarternion(robotOrn)))
    mu_list.append(vec_list)
    #phy_list.append(friction_list)

    time.sleep(0.01)

## Dump Pickle
with open('simulation_traject.pkl', 'wb') as f:
    pickle.dump(info_dict, f)

## Test 1: without friction setting
utils.dump2Tensor(xyz_list, 
                  ori_list, 
                  mu_list, 
                  save_filename = ['ts_input.pt', 'ts_output.pt'],
                  saveList = ['xyz_list.pkl', 'ori_list.pkl', 'mu_list.pkl'])

p.disconnect()
    
    