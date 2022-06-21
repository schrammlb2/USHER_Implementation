import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from pyquaternion import Quaternion

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -10)
useRealTimeSim = 0

p.setRealTimeSimulation(useRealTimeSim)  # either this
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("racecar/racecar.urdf")


def SetMotorTorqueById(robot_id, motor_id, torque):
    p.setJointMotorControl2(bodyIndex=robot_id,
                            jointIndex=motor_id,
                            controlMode=p.TORQUE_CONTROL,
                            force=torque)


def SetDesiredVelocityById(robot_id, motor_id, velocity, maxForce):
    p.setJointMotorControl2(bodyIndex=robot_id,
                            jointIndex=motor_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=velocity,
                            force=maxForce)


info_dict = {
    'jointId': {2: 'left_rear',
                3: 'right_rear',
                5: 'left_front',
                7: 'right_front'},
    'init_motorStatus': {2: {'controlMode': 0, 'targetVelocity': -.5, 'force': .5},
                         3: {'controlMode': 0, 'targetVelocity': -.5, 'force': .5},
                         5: {'controlMode': 0, 'targetVelocity': 0, 'force': .5},
                         7: {'controlMode': 0, 'targetVelocity': 0, 'force': .5}},
    'init_physical_para': {2: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0},
                           3: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0},
                           5: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0},
                           7: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0}},
    'global_GUI_para': {},
    'joint_GUI_para': {}}


####################################################
USE_DUMMY_TORQUE_CONTROL = True
####################################################

p.changeDynamics(robotId, -1, linearDamping=0, angularDamping=0)
for j in range(p.getNumJoints(robotId)):
    p.changeDynamics(robotId, j, linearDamping=0,
                     angularDamping=0)

for jid, _ in info_dict['jointId'].items():
    info_dict['joint_GUI_para'][jid] = {}

# Seperate Wheel: For each wheel, generate a slide bar to control velocity and force
# Global: Set same velocity and force to all wheel

for jid, para in info_dict['joint_GUI_para'].items():
    para["vecId"] = p.addUserDebugParameter(
        info_dict['jointId'][jid] + "_targetVelocity", -10, 10, info_dict['init_motorStatus'][jid]['targetVelocity'])  # info_dict['init_motorStatus'][jid]['targetVelocity']
for jid, para in info_dict['joint_GUI_para'].items():
    para["forceId"] = p.addUserDebugParameter(
        info_dict['jointId'][jid] + "_force", 0, 10, info_dict['init_motorStatus'][jid]['force'])  # info_dict['init_motorStatus'][jid]['force']

############# Initialization ##################
# Initialize Motor Status
if info_dict['init_motorStatus'] != {}:
    for jid, para in info_dict['init_motorStatus'].items():
        para['bodyUniqueId'] = robotId
        para['jointIndex'] = jid
        p.setJointMotorControl2(**para)

# Initialize Physical Parameters
if info_dict['init_physical_para'] != {}:
    for jid, para in info_dict['init_physical_para'].items():
        para['bodyUniqueId'] = robotId
        para['linkIndex'] = jid
        p.changeDynamics(**para)

################################################
# # Debug Text
# debugText = "TS: %.6d\nBase velocity = %.5f, angle = %.5f\nLeft Rear vec = %.5f, applied_torque = %.5f" % (
#     0, 0, 0, 0, 0)
# textId = p.addUserDebugText(debugText, [0, 0, .1], textColorRGB=[1, 0, 0])

flag = True
for i in range(1000000):

    for jid, _ in info_dict['jointId'].items():
        wheel = info_dict['joint_GUI_para'][jid]

        targetVelocity = p.readUserDebugParameter(
            wheel["vecId"])
        maxForce = p.readUserDebugParameter(wheel["forceId"])
        _, jtVec, _, jtTorque = p.getJointState(robotId, jid)
        #   tarV    actV    torque
        #   -10     -5      -max
        #   -2      -7      max
        #   10      -5      max
        ### Dummy Version of p.TORQUE_CONTROL
        # if maxForce == 0:
        #     applied_torque = 0
        #     SetMotorTorqueById(robotId, jid, 0)  # Set to Inactive
        if USE_DUMMY_TORQUE_CONTROL:
            if jtVec == targetVelocity: 
                ## Very uncommon for jtVec == targetVelocity when they are both float.
                ## Or a smaller torque when close to targetVelocity (How close?)
                applied_torque = 0
            elif jtVec < targetVelocity:
                applied_torque = maxForce
            else:
                applied_torque = -maxForce
            SetMotorTorqueById(robotId, jid, applied_torque)
        else:
            SetDesiredVelocityById(robotId, jid, targetVelocity, maxForce)


    lin, _ = p.getBaseVelocity(robotId)
    _, robotOri = p.getBasePositionAndOrientation(robotId)
    q = Quaternion(robotOri[3], robotOri[0], robotOri[1], robotOri[2])
    _, jtvec, _, jttorque = p.getJointState(robotId, 2)
    if jtvec > 0.09 and i > 10000 and flag:
        print(i, lin[0], jtvec)
        flag = False
        print()
    print("%.6d, %.5f, %.5f; %.5f, %.5f" %
          (i, lin[0], q.angle, jtvec, jttorque), end='\r')
    if not useRealTimeSim:
        p.stepSimulation()

    # debugText = "TS: %.6d\nBase velocity = %.5f, angle = %.5f\nLeft Rear vec = %.5f, applied_torque = %.5f" % (
    #     i, lin[0], q.angle, jtvec, jttorque)
    # p.addUserDebugText(debugText, [-1, -1, .1], textColorRGB=[
    #                    1, 0, 0], replaceItemUniqueId=textId)

print(lin[0])
p.disconnect()
