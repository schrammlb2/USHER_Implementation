
import pybullet as p
import time
import pybullet_data
from pyquaternion import Quaternion
import sys

if(len(sys.argv) != 2):
    print('Usage: python',sys.argv[0],'<urdf file>')
    sys.exit(1)

filepath = sys.argv[1]

############################################

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF("plane.urdf")

def _SetMotorTorqueById(robot_id, motor_id, torque):
    p.setJointMotorControl2(bodyIndex=robot_id,
                            jointIndex=motor_id,
                            controlMode=p.TORQUE_CONTROL,
                            force=torque)

def _SetMotorVelocityById(robot_id, motor_id, velocity, force):
    p.setJointMotorControl2(bodyIndex=robot_id,
                            jointIndex=motor_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity = velocity,
                            force=force)

car = p.loadURDF(filepath,[0,0,0], [0,0,0,1])
p.setGravity(0,0,-10) # m/s^2

wheelIds_dict={}
default_v = 100
wheelIds_dict["back_left"] = dict()
wheelIds_dict["back_right"] = dict()
wheelIds_dict["front_left"] = dict()
wheelIds_dict["front_right"] = dict()
wheelIds_dict["back_left"]["id"] = 0   ## Yellow -
wheelIds_dict["back_right"]["id"] = 1   ## Red + 
wheelIds_dict["front_left"]["id"] = 3  ## Blue -
wheelIds_dict["front_right"]["id"] = 2 ## Green +
wheelIds_dict["back_left"]["signal"] = -1   ## ->
wheelIds_dict["back_right"]["signal"] = +1   ## -< 
wheelIds_dict["front_left"]["signal"] = -1  ## ->
wheelIds_dict["front_right"]["signal"] = +1 ## -<
for _, wheel in wheelIds_dict.items():
    wheel["velocity"] = wheel["signal"] * default_v
    wheel["force"] = 0
    wheel["lateral_fricId"] = 0.5

##############################
useRealTimeSim = 0
seperate_wheel = False
GUI_friction = False

p.setRealTimeSimulation(useRealTimeSim)
##############################

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(car, -1, linearDamping=0, angularDamping=0)

for j in range(p.getNumJoints(car)):
  p.changeDynamics(car, j, linearDamping=0, angularDamping=0)


if not seperate_wheel:
    targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -200, 200, 0)
    maxForceSlider = p.addUserDebugParameter("maxForce", 0, 100, 10)
else:
    for key, wheel in wheelIds_dict.items():
        wheel["vecId"] = p.addUserDebugParameter(key + "wheelVelocity", -200, 200, 0)
        wheel["forceId"] = p.addUserDebugParameter(key + "maxForce", 0, 100, 10)

if GUI_friction:
    for key, wheel in wheelIds_dict.items():
        wheel["lateral_fricId"] = p.addUserDebugParameter(key + "fric", 0, 1, 0.5)


dynamic_list = ['mass', 'lateral_friction', 'inertia_diag', 'inertia_pos',
                'inertia_ori', 'restitution', 'rolling_fric', 'spinning_fric',
                'contact_damping', 'contact_stiffness']
for key, wheel in wheelIds_dict.items():
    info = p.getDynamicsInfo(car,wheel["id"])
    inf = dict(zip(dynamic_list, info))
    print(inf)

# for key, wheel in wheelIds_dict.items():
#     p.setJointMotorControl2(bodyUniqueId=car, jointIndex=wheel["id"], 
#                                 controlMode=p.VELOCITY_CONTROL, targetVelocity = wheel['signal'] * 1000, 
#                                 force = wheel["frictionForce"])

for i in range(1000000):
    if not seperate_wheel:
        targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
        maxForce = p.readUserDebugParameter(maxForceSlider)
    for key, wheel in wheelIds_dict.items():
        if seperate_wheel:
            targetVelocity = p.readUserDebugParameter(wheel["vecId"]) * wheel['signal']
            maxForce = p.readUserDebugParameter(wheel["forceId"])
        p.setJointMotorControl2(bodyUniqueId=car, jointIndex=wheel["id"], 
                                controlMode=p.VELOCITY_CONTROL, targetVelocity = targetVelocity * wheel['signal'], 
                                force = maxForce)
        if GUI_friction:
            tmp = p.readUserDebugParameter(wheel["lateral_fricId"])
            p.changeDynamics(bodyUniqueId=car,
                             linkIndex=wheel["id"], lateralFriction=tmp)
    lin, _ = p.getBaseVelocity(car)
    _, robotOri = p.getBasePositionAndOrientation(car)
    q = Quaternion(robotOri[3], robotOri[0], robotOri[1], robotOri[2])
    _, jtvec, _, jttorque = p.getJointState(car, 2)

    print("                         %.6d, %.5f, %.5f; %.5f, %.5f" %
          (i, lin[0], q.angle, jtvec, jttorque), end='\r')
    if not useRealTimeSim:
        p.stepSimulation()
        time.sleep(0.01)
    
