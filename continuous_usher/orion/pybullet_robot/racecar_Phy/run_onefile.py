import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from pyquaternion import Quaternion


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

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -10)
useRealTimeSim = 0

p.setRealTimeSimulation(useRealTimeSim)  # either this
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("racecar/racecar.urdf")

info_dict = {
    'jointId': {2: 'left_rear',
                3: 'right_rear',
                5: 'left_front',
                7: 'right_front'},
    'init_motorStatus': {2: {'controlMode': 0, 'targetVelocity': -.5, 'force': .5},
                         3: {'controlMode': 0, 'targetVelocity': 0, 'force': .5},
                         5: {'controlMode': 0, 'targetVelocity': 0, 'force': .5},
                         7: {'controlMode': 0, 'targetVelocity': 0, 'force': .5}},
    'init_physical_para': {2: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0},
                           3: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0},
                           5: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0},
                           7: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': 0.5, 'rollingFriction': 0, 'spinningFriction': 0}},
    'global_GUI_para': {},
    'joint_GUI_para': {}}


####################################################

global_wheel_control = True
change_friction_in_simulation = False
USE_DUMMY_TORQUE_CONTROL = False
    ## There are indeed performance difference between the dummy implementation and the true velocity control
    ## When setting single-wheel driven
    ## saying only motor [2] is set with targetVelocity = -0.5
####################################################

p.changeDynamics(robotId, -1, linearDamping=0, angularDamping=0)
for j in range(p.getNumJoints(robotId)):
    p.changeDynamics(robotId, j, linearDamping=0,
                     angularDamping=0)

for jid, _ in info_dict['jointId'].items():
    info_dict['joint_GUI_para'][jid] = {}

# Seperate Wheel: For each wheel, generate a slide bar to control velocity and force
# Global: Set same velocity and force to all wheel
if global_wheel_control:
    try:
        # Pick the first element from info_dict. Might get a key error if it's an empty dict
        key_1 = list(info_dict['init_motorStatus'].keys())[0]
        default_vec = info_dict['init_motorStatus'][key_1]['targetVelocity']
        default_force = info_dict['init_motorStatus'][key_1]['force']
    except:
        default_vec = 0
        default_force = 0
        print("Set default velocity as {} and force as {}".format(
            default_vec, default_force))
    finally:
        info_dict['global_GUI_para']['targetVelocitySlider'] = p.addUserDebugParameter(
            "global_targetVelocity", -100, 100, default_vec)
        info_dict['global_GUI_para']['maxForceSlider'] = p.addUserDebugParameter(
            "global_force", 0, 10, default_force)
else:
    for jid, para in info_dict['joint_GUI_para'].items():
        para["vecId"] = p.addUserDebugParameter(
            info_dict['jointId'][jid] + "_targetVelocity", -10, 10, info_dict['init_motorStatus'][jid]['targetVelocity'])  # info_dict['init_motorStatus'][jid]['targetVelocity']
    for jid, para in info_dict['joint_GUI_para'].items():
        para["forceId"] = p.addUserDebugParameter(
            info_dict['jointId'][jid] + "_force", 0, 10, info_dict['init_motorStatus'][jid]['force'])  # info_dict['init_motorStatus'][jid]['force']

# GUI_Friciton: Use Gui to change dynamics during simulations
if change_friction_in_simulation:
    for jid, para in info_dict['joint_GUI_para'].items():
        para["lateral_fricId"] = p.addUserDebugParameter(
            info_dict['jointId'][jid] + "_fric", 0, 100, 0.5)

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
# # Debug Text: Will shown on GUI but will make simulation much slower
# debugText = "TS: %.6d\nBase velocity = %.5f, angle = %.5f\nLeft Rear vec = %.5f, applied_torque = %.5f" % (0, 0, 0, 0, 0)
# textId = p.addUserDebugText(debugText, [0, 0, .1], textColorRGB=[1, 0, 0])

flag = True
for i in range(1000000):
    if global_wheel_control:
        targetVelocity = p.readUserDebugParameter(
            info_dict['global_GUI_para']['targetVelocitySlider'])
        maxForce = p.readUserDebugParameter(
            info_dict['global_GUI_para']['maxForceSlider'])
    for jid, _ in info_dict['jointId'].items():
        wheel = info_dict['joint_GUI_para'][jid]
        if not global_wheel_control:
            targetVelocity = p.readUserDebugParameter(
                wheel["vecId"])
            maxForce = p.readUserDebugParameter(wheel["forceId"])

            _, jtVec, _, jtTorque = p.getJointState(robotId, jid)
        
        #######################
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
        else: ## Velocity Control
            SetDesiredVelocityById(robotId, jid, targetVelocity, maxForce)

        #######################
        if change_friction_in_simulation:
            tmp = p.readUserDebugParameter(wheel["lateral_fricId"])
            p.changeDynamics(bodyUniqueId=robotId,
                             linkIndex=jid, lateralFriction=tmp)

    lin, _ = p.getBaseVelocity(robotId)
    _, robotOri = p.getBasePositionAndOrientation(robotId)
    q = Quaternion(robotOri[3], robotOri[0], robotOri[1], robotOri[2])
    _, jtvec, _, jttorque = p.getJointState(robotId, 2)
    # if jtvec > 0.09 and i > 10000 and flag:
    #     print(i, lin[0], jtvec)
    #     flag = False
    #     print()
    print("                         %.6d, %.5f, %.5f; %.5f, %.5f" %
          (i, lin[0], q.angle, jtvec, jttorque), end='\r')
    if not useRealTimeSim:
        p.stepSimulation()
        time.sleep(0.01)

    # debugText = "TS: %.6d\nBase velocity = %.5f, angle = %.5f\nLeft Rear vec = %.5f, applied_torque = %.5f" % (i, lin[0], q.angle, jtvec, jttorque)
    # p.addUserDebugText(debugText, [-1, -1, .1], textColorRGB=[
    #                    1, 0, 0], replaceItemUniqueId=textId)

print(lin[0])
p.disconnect()
