import time
import pybullet_data
import pybullet as p
import os

def SetDesiredVelocityById(robot_id, motor_id, velocity, maxForce):
    p.setJointMotorControl2(bodyIndex=robot_id,
                            jointIndex=motor_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=velocity,
                            force=maxForce)

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -10)

useRealTimeSim = 1

p.setRealTimeSimulation(useRealTimeSim)  # either this
p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
car = p.loadURDF("CAR_v000.urdf")

wheels = [35,2,24,13]
active_wheels = [35,2,24,13]
left_rear = 35; right_rear = 2; left_front = 24; right_front = 13

forward_signal = [1, -1, 1, -1]

# p.changeDynamics(car, -1, linearDamping=0, angularDamping=0)
# for j in range(p.getNumJoints(car)):
#     p.changeDynamics(car, j, linearDamping=0,
#                      angularDamping=0)

for wheel in wheels:
    p.setJointMotorControl2(
        car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

for wheel in wheels:
    if wheel in active_wheels:
        p.changeDynamics(car, wheel, lateralFriction=0.5)
    else:
        p.changeDynamics(car, wheel, lateralFriction=0.01)

for i in range(p.getNumJoints(car)):
    print(p.getDynamicsInfo(car, i))

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -100, 100, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 50, 5)

#####################################################################

while (True):
    maxForce = p.readUserDebugParameter(maxForceSlider)
    targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
    # print(targetVelocity)

    for wheel, signal in zip(wheels, forward_signal):
        if wheel in active_wheels:
            SetDesiredVelocityById(car, wheel, targetVelocity * signal, maxForce)

    if (useRealTimeSim == 0):
        p.stepSimulation()

    _, jtVec, _, jtTorque = p.getJointState(car, 2)
    print("                                                          Wheel Velocity: {:10.8}, Applied Torque: {:10.8}".format(jtVec, jtTorque), end="\r")
    #time.sleep(0.01)
