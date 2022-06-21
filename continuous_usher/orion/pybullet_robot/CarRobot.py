import numpy as np
import pybullet as p
import pybullet_data
import time
import os

import torch

import pickle
from pyquaternion import Quaternion

class CarRobot():
    """
    Single Bullet Engine Manager:
    * Initialize bullet
    * Load Plane and Robot
    * Apply control signals (From OBS)
    * Collect observations
    * Dump files
    """
    def __init__(self, clientP, ObsManager, setting):
        self.observations = []
        self.clientP = clientP
        self.ObsManager = ObsManager

        ## Read Settings (from yamls)
        self.setting = setting
        self.basename = setting['basename']
        self.frames = setting['frames_before_trim']
        self.robot_file = setting['robotURDF']
        self.start_pos = setting['startPosition']
        self.start_ori = setting['startOrientation']
        self.num_trim = setting['n_trim']
        
        ## Extra setting from BulletClientManger(paras)
        self.output_dir = setting['output_dir']
        self.debug = setting['debug']
        self.GUI = setting['GUI']
        
        ## Initialize
        self._initialize()

    def _initialize(self):
        self._stepCounter = 0

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
    
    def _loadRobot(self, robot_file, start_pos, start_ori):
        self.robotId = p.loadURDF(robot_file, 
                             start_pos, start_ori)
        return self.robotId

    def _initDynamicDumpings(self):
        p.changeDynamics(self.robotId, -1, linearDamping=0, angularDamping=0)
        for j in range(p.getNumJoints(self.robotId)):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            #info = p.getJointInfo(self.robotId, j)
    
    def _initDebugParaGUI(self):
        self.ObsManager._initDebugParaGUI()
        
    def _initMotorStatus(self, startMotorStatus):
        self.ObsManager._initMotorStatus(startMotorStatus)

    def _initPhysicalParas(self, physicalParas):
        self.ObsManager._initPhysicalParas(physicalParas)

    def printDynamics(self):
        info = p.getDynamicsInfo(self.robotId, -1)
        print(info)
        for j in range(p.getNumJoints(self.robotId)):
            info = p.getDynamicsInfo(self.robotId, j)
            print(info)

    def _get_info(self):
        return [self._stepCounter] + self.ObsManager._getInfo()

    def dumpObservation(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.observations, f)

    def _trimObservation(self, num):
        """
        When loading robot, the z axis need adjustment in the first few steps
        """
        if num > 0  and len(self.observations) > num+10: ## At least left 10 records?
            #self.observations = self.observations[num:]
            del self.observations[:num]

    def run(self):
        # print("Starting {} epoch".format(epo))
        self._initialize()
        self.ObsManager.robotId  = self._loadRobot(robot_file=self.robot_file,
                                                   start_pos=self.start_pos, 
                                                   start_ori=self.start_ori)
        #self._initDebugParaGUI()
        self._initDynamicDumpings()
        self._initMotorStatus(self.setting['startMotorStatus'])
        self._initPhysicalParas(self.setting['startDynamics'])

        self.printDynamics()

        #time.sleep(3)
        
        action_frame = list(self.setting['contorl_signal'])
        
        for i in range(self.frames):
            if i in action_frame:
                control_signal_i = self.setting['contorl_signal'][i]
                #print("Frame {} get Control Signal {}\n".format(i, control_signal_i))
                self.ObsManager._assignControlSignal(control_signal_i)
            info = self._step()
            if self.debug:
                print(info)

        self._trimObservation(self.num_trim)

        if self.output_dir is not None:
            filename = os.path.join(self.output_dir, 'obs_' + self.basename + '.pkl')
            self.dumpObservation(filename)

        if self.debug:
            print("Length of output: {}".format(len(self.observations)))
            print("observation samples: ")
            print(self.observations[-1])
        
        if self.GUI:
            p.disconnect()