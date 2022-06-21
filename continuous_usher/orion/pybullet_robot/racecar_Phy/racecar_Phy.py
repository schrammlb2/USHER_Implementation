import time
import pybullet_data
import pybullet as p
import os

# (0.1, 0.5, (6.666666666666667e-08, 6.666666666666667e-08, 6.666666666666667e-08), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.1, 0.5, (6.666666666666667e-08, 6.666666666666667e-08, 6.666666666666667e-08), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (4.0, 0.5, (2.6666666666666664e-06, 2.6666666666666664e-06, 2.6666666666666664e-06), (0.1477, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.34055, 0.5, (0.0003926825134593656, 0.0003926825134593656, 0.0006377366078238682), (0.0, 0.0, -0.0225), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.34055, 0.5, (0.0003926825134593656, 0.0003926825134593656, 0.0006377366078238682), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.1, 0.5, (6.666666666666667e-08, 6.666666666666667e-08, 6.666666666666667e-08), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.34055, 0.5, (0.0003926825134593656, 0.0003926825134593656, 0.0006377366078238682), (0.0, 0.0, -0.0225), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.1, 0.5, (6.666666666666667e-08, 6.666666666666667e-08, 6.666666666666667e-08), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.34055, 0.5, (0.0003926825134593656, 0.0003926825134593656, 0.0006377366078238682), (0.0, 0.0, 0.0225), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.13, 0.5, (0.0002166666666666667, 0.0002166666666666667, 0.0002166666666666667), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (1e-05, 0.5, (2.6270833333333336e-08, 1.6575000000000005e-09, 2.6428333333333336e-08), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (1e-05, 0.5, (6.666666666666667e-12, 6.666666666666667e-12, 6.666666666666667e-12), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (1e-05, 0.5, (6.666666666666667e-12, 6.666666666666667e-12, 6.666666666666667e-12), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)

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
car = p.loadURDF("racecar/racecar.urdf")

wheels = [2, 3, 5, 7]
left_rear = 2; right_rear = 3; left_front = 5; right_front = 7

forward_signal = [1, 1, 1, 1]

# p.changeDynamics(car, -1, linearDamping=0, angularDamping=0)
# for j in range(p.getNumJoints(car)):
#     p.changeDynamics(car, j, linearDamping=0,
#                      angularDamping=0)

for wheel in wheels:
    p.setJointMotorControl2(
        car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

for wheel in wheels:
    if wheel in [2]:
        p.changeDynamics(car, wheel, lateralFriction=0.5)
    else:
        p.changeDynamics(car, wheel, lateralFriction=0)

print(p.getDynamicsInfo(car, -1))
for i in range(p.getNumJoints(car)):
    print(p.getDynamicsInfo(car, i))

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -5, 5, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 50, 5)

#####################################################################

while (True):
    maxForce = p.readUserDebugParameter(maxForceSlider)
    targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
    # print(targetVelocity)

    for wheel, signal in zip(wheels, forward_signal):
        if wheel in wheels:
            SetDesiredVelocityById(car, wheel, targetVelocity * signal, maxForce)

    if (useRealTimeSim == 0):
        p.stepSimulation()

    _, jtVec, _, jtTorque = p.getJointState(car, 2)
    print("                                                          Wheel Velocity: {:10.8}, Applied Torque: {:10.8}".format(jtVec, jtTorque), end="\r")
    #time.sleep(0.01)
