import numpy as np
import pybullet as p
import pybullet_data
import time
import os

import torch

import pickle
from pyquaternion import Quaternion
#import pybullet_robot.CAR_v000.utils as utils
import NN_model.data_utils.BulletObservations as nnBulletObs


class Carbot():
    """
    Bullet Engine Manager:
    * Initialize bullet
    * Load Plane and Robot
    * Apply control signals (From OBS)
    * Collect observations
    * Dump files
    """
    def __init__(self, ObsManager, epochs = 1, frames = 20, num_trim =-1, render=False):
        self.observations = []
        self.ObsManager = ObsManager
        self.frames = frames
        self.epochs = epochs ## TODO
        self.num_trim = num_trim
        self.render = render

    def _step(self):
        p.stepSimulation()
        ## Observation: Data colleciton of List
        self.observations.append(self._compute_observation())
        ## Info: msg that will be printed each timestep
        info = self._get_info()
        self._stepCounter += 1
        return info

    def _compute_observation(self):
        obs = self.ObsManager._generateObservations()
        return obs

    
    def _reset(self):
        if (self.render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        p.setPhysicsEngineParameter(numSolverIterations=10)

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
        return self.robotId

    def _initDynamicDumpings(self):
        p.changeDynamics(self.robotId, -1, linearDamping=0, angularDamping=0)
        for j in range(p.getNumJoints(self.robotId)):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)
        
    def _initDebugParaGUI(self):
        self.ObsManager._initDebugParaGUI()

    def _get_info(self):
        return [self._stepCounter] + self.ObsManager._getInfo()
    
    def GetObservations(self):
        return self.observations

    # def change_control_signal(self, signal = [1,-1,1,-1], velocity = 200, friction_force = [0.1, 0.1, 0.1, 0.1]):
    #     for i in range(4):
    #         assert signal[i] in [-1, 0, 1], "Unvalid signal."
    #     ## BL(+), BR(-), FL(+), FR(-)
    #     wheel_key = list(self.ObsManager.wheelIds_dict.keys())
    #     apply_dict = dict(zip(wheel_key, signal))

    #     applyf_dict = dict(zip(wheel_key, friction_force))

    #     for key, wheel in self.ObsManager.wheelIds_dict.items():
    #         wheel["velocity"] = apply_dict[key] * velocity
    #         wheel['frictionForce'] = applyf_dict[key]

    # ## TODO: No effect? Have to implement setJointMotor instead....
    # def change_friction(self, lateral_frcition = [0.5, 0.5, 0.5, 0.5]):
    #     wheel_key = list(self.ObsManager.wheelIds_dict.keys())
    #     apply_dict = dict(zip(wheel_key, lateral_frcition))
    #     for key, wheel in self.ObsManager.wheelIds_dict.items():
    #         p.changeDynamics(self.robotId, wheel["id"], angularDamping=apply_dict[key])

    def dumpObservation(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.observations, f)

    def _trimObservation(self, num=20):
        """
        When loading robot, the z axis need adjustment in the first few steps
        """
        if len(self.observations) > num+10: ## At least left 10 records?
            self.observations = self.observations[num:]

    def run(self, filedir=None, debug = False):
        # print("Starting {} epoch".format(epo))
        self._reset()
        self.ObsManager.robotId  = self._loadRobot(start_pos = [0,0,0.01], 
                                                start_ori = [0,0,0,1])
        
        self._initDebugParaGUI()
        self._initDynamicDumpings()

        time.sleep(3)
        
        # BL, BR, FL, FR
        # #self.change_control_signal(signal=[1,-1,1,-1], friction_force = [1e-04, 1e-04, 1e-04, 1e-04])
        # self.change_control_signal(signal=[1,-1,1,-1])
        # self._assign_motor()
        # #self.change_friction([0.1,.5,.5,0.1])
        
        for i in range(self.frames):
            self.ObsManager._changeDynamics()
            info = self._step()
            if debug:
                print(info)

        self._trimObservation(self.num_trim)

        if filedir is not None:
            filename = os.path.join(filedir, 'traj_%.4d.pkl' % (self.epochs))
            self.dumpObservation(filename)

        if debug:
            print("Length of output: {}".format(len(self.observations)))
            print("observation samples: ")
            print(self.observations[-1])
        p.disconnect()

        return filename
            

if __name__ == "__main__":
    obs2 = nnBulletObs.getBulletObs_2()
    bullet = Carbot(render=True, ObsManager=obs2)
    bullet.run(debug=True)