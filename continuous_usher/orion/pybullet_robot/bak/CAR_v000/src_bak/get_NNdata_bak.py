import numpy as np
import pybullet as p
import pybullet_data
import time
import os

import torch

import pickle
from pyquaternion import Quaternion
import pybullet_robot.CAR_v000.utils as utils


def generateComponentDict():
    wheelIds_dict={}
    default_v = 200
    wheel_friction_froce = 1
    wheelIds_dict["back_left"] = dict()
    wheelIds_dict["back_right"] = dict()
    wheelIds_dict["front_left"] = dict()
    wheelIds_dict["front_right"] = dict()
    wheelIds_dict["back_left"]["id"] = 35   ## ->
    wheelIds_dict["back_right"]["id"] = 2   ## -< 
    wheelIds_dict["front_left"]["id"] = 24  ## ->
    wheelIds_dict["front_right"]["id"] = 13 ## -<
    wheelIds_dict["back_left"]["velocity"] = default_v   ## ->
    wheelIds_dict["back_right"]["velocity"] = -default_v   ## -< 
    wheelIds_dict["front_left"]["velocity"] = default_v  ## ->
    wheelIds_dict["front_right"]["velocity"] = -default_v ## -<
    for _, wheel in wheelIds_dict.items():
        wheel["frictionForce"] = wheel_friction_froce

    controlTargetIds_dict={}
    controlTargetIds_dict["Neck_Motor"] = dict()
    controlTargetIds_dict["Bot_Motor"] = dict()
    controlTargetIds_dict["Neck_Motor"]["id"] = 53
    controlTargetIds_dict["Bot_Motor"]["id"] = 51

    return wheelIds_dict, controlTargetIds_dict


class Carbot():
    def __init__(self, epochs = 20, obs_set = '1', render=False):
        self.observation = []
        self.obs_set = obs_set

        self.wheelIds_dict, self.controlTargetIds_dict = generateComponentDict()

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        p.setPhysicsEngineParameter(numSolverIterations=10)

        self.epochs = epochs

    def _step(self):
        p.stepSimulation()
        ## Observation: Data colleciton of List
        self.observation.append(self._compute_observation())
        ## Info: msg that will be printed each timestep
        info = self._get_info()
        self._stepCounter += 1
        return info

    def getFrictionTuple(self):
        fric_tuple = ()
        for _, wheel in self.wheelIds_dict.items():
            jointInfoList = p.getDynamicsInfo(self.robotId, wheel["id"])
            fric_tuple += (jointInfoList[1],) # lateral_friction (6: rolling, 7: spinning)
        return fric_tuple

    def getVelocitySignal(self):
        vec_tuple = ()
        for _, wheel in self.wheelIds_dict.items():
            vec_tuple += (wheel["velocity"],)
        return vec_tuple

    def _compute_observation(self):
        obs = []
        robotPos, robotOrn = p.getBasePositionAndOrientation(self.robotId)
        robotLinearVec, robotAngVEc = p.getBaseVelocity(self.robotId) ## [[xdot,ydot,zdot], [wx,wy,wz]]
        vec_tuple = self.getVelocitySignal()
        fric_tuple = self.getFrictionTuple()

        ## Set 1: Classic Position, Orientation, velocity
        if self.obs_set == '1':
            obs = [robotPos, robotOrn, vec_tuple, fric_tuple]
        ## Set 2: Linear x,y, Angular z, one_hot_12 control signal
        elif self.obs_set == '2':
            ## Take spaces....
            vec_onehot = utils.tuple_to_one_hot(vec_tuple)
            linear_xydot = robotLinearVec[0:2]
            angular_wz = robotAngVEc[2]
            obs = [robotPos, linear_xydot, angular_wz, vec_onehot]
        elif self.obs_set == '3':
            ## Take spaces....
            vec_onehot = utils.tuple_to_one_hot(vec_tuple)
            linear_xydot = robotLinearVec[0:2]
            angular_wz = robotAngVEc[2]
            robotPos = robotPos[0:2]
            obs = [robotPos, linear_xydot, angular_wz, vec_onehot]
        ## TODO
        else:
            ## 3, 4, 3, 3, 4,
           raise NotImplementedError

        ## Get Back Data
        ##  rP, rO, rLC, rAC, vec = zip(*dat)
        return obs

    
    def _reset(self):
        self._stepCounter = 0
        p.resetSimulation()        
        p.setGravity(0,0,-10) # m/s^2
        p.setTimeStep(0.01) # sec
        plane_id = p.loadURDF("plane.urdf")
    
    def _loadRobot(self, start_pos = [0,0,0], start_ori = [0,0,0,1]):
        # start_pos = [0, 0, 0.01]
        # start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))
        self.robotId = p.loadURDF(os.path.join(path, "CAR_v000.urdf"), 
                             start_pos, start_ori)

    def _initDynamics(self):
        p.changeDynamics(self.robotId, -1, linearDamping=0, angularDamping=0)
        for j in range(p.getNumJoints(self.robotId)):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)
    
    def _assign_motor(self):
        for _, wheel in self.wheelIds_dict.items():
            p.setJointMotorControl2(bodyUniqueId=self.robotId, 
                                    jointIndex=wheel["id"], 
                                    controlMode=p.VELOCITY_CONTROL, 
                                    targetVelocity = wheel["velocity"], 
                                    force = wheel["frictionForce"])

        for _, controlTarget in self.controlTargetIds_dict.items():
            p.setJointMotorControl2(self.robotId, 
                                    controlTarget["id"], 
                                    p.POSITION_CONTROL, 
                                    0, # Stay default 
                                    force=1000)

    def _get_info(self):
        robotPos, _ = p.getBasePositionAndOrientation(self.robotId)
        robotLinearVec, robotAngVEc = p.getBaseVelocity(self.robotId)
        #fric_t = self.getFrictionTuple()
        vec_t = self.getVelocitySignal()
        return [self._stepCounter, robotLinearVec, vec_t]


    def change_control_signal(self, signal = [1,-1,1,-1], velocity = 200, friction_force = [0.1, 0.1, 0.1, 0.1]):
        for i in range(4):
            assert signal[i] in [-1, 0, 1], "Unvalid signal."
        ## BL(+), BR(-), FL(+), FR(-)
        wheel_key = list(self.wheelIds_dict.keys())
        apply_dict = dict(zip(wheel_key, signal))

        applyf_dict = dict(zip(wheel_key, friction_force))

        for key, wheel in self.wheelIds_dict.items():
            wheel["velocity"] = apply_dict[key] * velocity
            wheel['frictionForce'] = applyf_dict[key]

    ## TODO: No effect? Have to implement setJointMotor instead....
    def change_friction(self, lateral_frcition = [0.5, 0.5, 0.5, 0.5]):
        wheel_key = list(self.wheelIds_dict.keys())
        apply_dict = dict(zip(wheel_key, lateral_frcition))
        for key, wheel in self.wheelIds_dict.items():
            p.changeDynamics(self.robotId, wheel["id"], angularDamping=apply_dict[key])

    def dumpObservation(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.observation, f)

    def _trimObservation(self, num=20):
        """
        When loading robot, the z axis need adjustment in the first few steps
        """
        if len(self.observation) > num+10: ## At least left 10 records?
            self.observation = self.observation[num:len(self.observation)]        

    def run(self, filepath=None, debug = False):
        self._reset()
        self._loadRobot(start_pos = [0,0,0.01], start_ori = [0,0,0,1])
        self._initDynamics()  
        # BL, BR, FL, FR
        #self.change_control_signal(signal=[1,-1,1,-1], friction_force = [1e-04, 1e-04, 1e-04, 1e-04])
        self.change_control_signal(signal=[1,-1,1,-1])
        #self.change_friction([0.1,.5,.5,0.1])

        for i in range(self.epochs):
            self._assign_motor()
            info = self._step()
            if debug:
                print(info)
        
        self._trimObservation(20)

        if filepath is not None:
            self.dumpObservation(filepath)

        if debug:
            print("observation samples: ")
            print(self.observation[-1])
        p.disconnect()
            

if __name__ == "__main__":
    bullet = Carbot(render=True, obs_set='2')
    bullet.run(debug=True)