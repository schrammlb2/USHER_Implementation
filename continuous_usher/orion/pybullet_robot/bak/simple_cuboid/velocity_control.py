import pybullet as p
import time
import pybullet_data
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF("plane.urdf")

def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                force=torque)

humanoid = p.loadURDF("simple_cuboid.urdf",[0,0,0], [0,0,0,1])

p.setGravity(0,0,-10) # m/s^2
p.setTimeStep(0.01)

wheelIds_dict={}
default_v = 200
wheel_friction_force = 100
wheelIds_dict["yellow"] = dict()
wheelIds_dict["red"] = dict()
wheelIds_dict["blue"] = dict()
wheelIds_dict["green"] = dict()
wheelIds_dict["yellow"]["id"] = 0   ## Yellow -
wheelIds_dict["red"]["id"] = 1   ## Red + 
wheelIds_dict["blue"]["id"] = 3  ## Blue -
wheelIds_dict["green"]["id"] = 2 ## Green +
wheelIds_dict["yellow"]["signal"] = +1   ## ->
wheelIds_dict["red"]["signal"] = -1   ## -< 
wheelIds_dict["blue"]["signal"] = +1  ## ->
wheelIds_dict["green"]["signal"] = -1 ## -<
for _, wheel in wheelIds_dict.items():
    wheel["velocity"] = wheel["signal"] * default_v
    wheel["frictionForce"] = wheel_friction_force
    wheel["lateral_fric"] = 0.5
    wheel["spinning_fric"] = 0
    wheel["rolling_fric"] = 0


p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)

for j in range(p.getNumJoints(humanoid)):
    p.changeDynamics(humanoid, j, linearDamping=0, angularDamping=0)

for key, wheel in wheelIds_dict.items():
    wheel["lateral_fricId"] = p.addUserDebugParameter(key + "lateral_friction", 0, 50, .5)
for key, wheel in wheelIds_dict.items():
    wheel["spinning_fricId"] = p.addUserDebugParameter(key + "spinning_friction", 0, 50, 0)
for key, wheel in wheelIds_dict.items():
    wheel["rolling_fricId"] = p.addUserDebugParameter(key + "rolling_friction", 0, 50, 0)
vecId = p.addUserDebugParameter("targetVeclocity", -2000, 2000, default_v)
# lateral_fricId = p.addUserDebugParameter("lateral_friction", 0, 50, .5)
# spinning_fricId = p.addUserDebugParameter("spinning_friction", 0, 50, 0)
# rolling_fricId = p.addUserDebugParameter("rolling_friction", 0, 50, 0)
vecForceId = p.addUserDebugParameter("force", 0, 1000, 10)

p.setRealTimeSimulation(0)
dynamic_list = ['mass', 'lateral_friction', 'inertia_diag', 'inertia_pos',
                'inertia_ori', 'restitution', 'rolling_fric', 'spinning_fric',
                'contact_damping', 'contact_stiffness']
for key, wheel in wheelIds_dict.items():
    info = p.getDynamicsInfo(humanoid,wheel["id"])
    inf = dict(zip(dynamic_list, info))
    print(inf)

targetVec = p.readUserDebugParameter(vecId)
maxForce = p.readUserDebugParameter(vecForceId)
# lateral_fric = p.readUserDebugParameter(lateral_fricId)

print(wheelIds_dict)
for key, wheel in wheelIds_dict.items():
    p.setJointMotorControl2(bodyUniqueId=humanoid, jointIndex=wheel["id"], 
                                controlMode=p.VELOCITY_CONTROL, targetVelocity = wheel["velocity"], 
                                force = 0)

time.sleep(3)

while(1):

    for key, wheel in wheelIds_dict.items():
        # targetVec = p.readUserDebugParameter(vecId)
        # maxForce = p.readUserDebugParameter(vecForceId)
        # p.setJointMotorControl2(bodyUniqueId=humanoid, jointIndex=wheel["id"], 
        #                         controlMode=p.VELOCITY_CONTROL, targetVelocity = targetVec, 
        #                         force = maxForce)
        lateral_friction = p.readUserDebugParameter(wheel["lateral_fricId"])
        spinning_friction = p.readUserDebugParameter(wheel["spinning_fricId"])
        rolling_friction = p.readUserDebugParameter(wheel["rolling_fricId"])
        p.changeDynamics(humanoid, wheel["id"], lateralFriction=lateral_friction,spinningFriction=spinning_friction,rollingFriction=rolling_friction)
    #robotPos, robotOri = p.getBasePositionAndOrientation(humanoid)
    #print(robotPos, robotOri)
    p.stepSimulation()
    #time.sleep(0.01)
    
