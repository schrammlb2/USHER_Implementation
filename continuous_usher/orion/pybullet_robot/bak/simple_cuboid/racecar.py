import time
import pybullet_data
import pybullet as p
import os

# (32.93232, 0.5, (0.19641933746961232, 0.1120796563064445, 0.22136007876597413), (0.0, 0.0, 0.06), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.7102512671235804, 0.5, (0.0005152285037549084, 0.0003431475393776207, 0.000343014816520304), (-0.016000000000000014, 1.3877787807814457e-17, -6.938893903907228e-18), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.7102512671235804, 0.5, (0.0005152285037549084, 0.0003431475393776207, 0.000343014816520304), (0.015999999999999986, 1.3877787807814457e-17, -6.938893903907228e-18), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.7102512671235804, 0.5, (0.0005152283873973919, 0.0003431475393776207, 0.0003430147001627875), (0.015999999999999986, 2.7755575615628914e-17, 2.2551405187698492e-17), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)
# (0.7102512671235804, 0.5, (0.0005152283873973919, 0.0003431475393776207, 0.0003430147001627875), (-0.016000000000000014, 2.7755575615628914e-17, 1.9081958235744878e-17), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, -1.0, -1.0, 2)


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
car = p.loadURDF("simple_cuboid.urdf")

wheels = [0, 1, 2, 3]
active_wheels = [0, 1, 2, 3]
#color = ['yellow_BR', 'red_BL', 'green_FL', 'blue_FR']
yellow = 0; red = 1; green = 2; blue = 3

forward_signal = [-1, 1, 1, -1]

# p.changeDynamics(car, -1, linearDamping=0, angularDamping=0)
# for j in range(p.getNumJoints(car)):
#     p.changeDynamics(car, j, linearDamping=0,
#                      angularDamping=0)

for wheel in wheels:
    p.setJointMotorControl2(
        car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

for wheel in wheels:
    if wheel in [2]:
        p.changeDynamics(car, wheel, lateralFriction=0.001)
    else:
        p.changeDynamics(car, wheel, lateralFriction=0.5)


print(p.getDynamicsInfo(car, -1))
for i in range(p.getNumJoints(car)):
    print(p.getDynamicsInfo(car, i))

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -100, 100, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 100, 50)

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
