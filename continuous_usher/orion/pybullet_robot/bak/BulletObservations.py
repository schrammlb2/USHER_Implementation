import pybullet as p
import numpy as np

import NN_model.data_utils.utils as utils

class BulletObservations(object):
    """
    1. Called by CarRobot():
        _init*()
        _initDynamicDumpings()
        _initMotorStatus()
        _initPhysicalParas()

        if GUI:
            _initDebugParaGUI()

        for i in range(epochs):
                if i in action_frames:
                    _assignControlSignal()
                if GUI:
                    _changeGUIParas()

    2. Called by NN data:
        _getDataset()
    """
    def __init__(self, robotId):
        raise NotImplementedError

    def _generateObservations(self):
        raise NotImplementedError

    def _getInfo(self):
        """
        if debug=True:
            print this returned value for each `p.stepSimulation`
        """
        return []

    def _assignControlSignal(self):
        return

    def _initMotorStatus(self, startMotorStatus):
        return
    
    def _initPhysicalParas(self, physicalParas):
        return

    ## if GUI
    def _initDebugParaGUI(self):
        return
    
    def _changeGUIParas(self):
        return

    ## NN get data
    def _getDataset(self, observations, getDelta = False, debug = False):
        raise NotImplementedError


class SimpleObs(BulletObservations):
    def __init__(self, robotId=None):
        self.robotId = robotId
        self.wheelIds_dict = self._componentDict()
    
    def _componentDict(self):
        wheelIds_dict={}
        default_v = 400
        wheel_friction_froce = 1
        wheelIds_dict["back_left"] = dict()
        wheelIds_dict["back_right"] = dict()
        wheelIds_dict["front_left"] = dict()
        wheelIds_dict["front_right"] = dict()
        wheelIds_dict["back_left"]["id"] = 0   ## ->
        wheelIds_dict["back_right"]["id"] = 1   ## -< 
        wheelIds_dict["front_left"]["id"] = 3  ## ->
        wheelIds_dict["front_right"]["id"] = 2 ## -<
        wheelIds_dict["back_left"]["velocity"] = default_v   ## ->
        wheelIds_dict["back_right"]["velocity"] = -default_v   ## -< 
        wheelIds_dict["front_left"]["velocity"] = default_v  ## ->
        wheelIds_dict["front_right"]["velocity"] = -default_v ## -<
        for _, wheel in wheelIds_dict.items():
            wheel["frictionForce"] = wheel_friction_froce
        return wheelIds_dict

    def _assign_motor(self):
        for _, wheel in self.wheelIds_dict.items():
            p.setJointMotorControl2(bodyUniqueId=self.robotId, 
                                    jointIndex=wheel["id"], 
                                    controlMode=p.VELOCITY_CONTROL, 
                                    targetVelocity = wheel["velocity"], 
                                    force = wheel["frictionForce"])
    
    def _initMotorStatus(self, startMotorStatus):
        for joint, status in startMotorStatus.items():
            for key, value in status.items():
                self.wheelIds_dict[joint][key] = value

    def _changeControlSignal(self, contorl_signal):
        ## contorl_signal = {'back_left': 400, 'back_right': -400, 'front_left': -400, 'front_right': 400}
        for joint, velocity in contorl_signal.items():
            self.wheelIds_dict[joint]['velocity'] = velocity            

    def _assignControlSignal(self,contorl_signal=None):
        #self._changeGUIParas()
        if contorl_signal is not None:
            self._changeControlSignal(contorl_signal)
        self._assign_motor()

    def _initPhysicalParas(self, physicalParas):
        ## TODO: Not validate yet. Yaml also not changed accordingly yet.
        ## Design as physicalParas = {'back_left': {'lateralFriction': 0.5, 'spinningFriction': 0.5}, ...}
        for joint, paras in physicalParas.items():
            ## TODO: Tidy parameter dict
            tmpDict = paras
            tmpDict['bodyUniqueId'] = self.robotId
            tmpDict['linkIndex'] = self.wheelIds_dict[joint]['id']
            ## TODO: Validate other items in tmpDict
            p.changeDynamics(**tmpDict)

###############

    def getVelocitySignal(self):
        vec_tuple = ()
        for _, wheel in self.wheelIds_dict.items():
            vec_tuple += (wheel["velocity"],)
        return vec_tuple

    def _generateObservations(self):
        robotPos, _ = p.getBasePositionAndOrientation(self.robotId)
        robotLinearVec, robotAngVEc = p.getBaseVelocity(self.robotId) ## [[xdot,ydot,zdot], [wx,wy,wz]]
        vec_tuple = self.getVelocitySignal()
        rP2 = robotPos[0:2]
        linear_xydot = robotLinearVec[0:2]
        angular_wz = robotAngVEc[2]
        obs = [rP2, linear_xydot, angular_wz, vec_tuple]
        return obs

    def _getDeltaDat(self, x_data, y_data):
        x_data, y_data = utils.getDeltaDataset(x_data, y_data)
        return x_data, y_data
        
    def _getDataset(self, observations, getDelta = False, debug = False):
        """
        Unwrap Method for observations -> [pos, linearV, angularV]
        """
        if type(observations) == str:
            observations = utils.loadpkl(observations)
        
        n = len(observations)
        tidy_dat = [np.asarray(x).reshape(n,-1) for x in zip(*observations)]

        rP2, linear_xydot, angular_wz, vec = tidy_dat
        x_data = np.hstack((rP2, linear_xydot, angular_wz, vec))
        y_data = np.hstack((rP2, linear_xydot, angular_wz))
        
        if getDelta:
            x_data, y_data = self._getDeltaDat(x_data, y_data)

        if debug:
            print(x_data[-1])
            print(y_data[-1])
        return x_data, y_data

    def _getInfo(self):
        robotPos, _ = p.getBasePositionAndOrientation(self.robotId)
        robotLinearVec, robotAngVEc = p.getBaseVelocity(self.robotId)
        vec_t = self.getVelocitySignal()
        return [robotPos, robotLinearVec, robotAngVEc, vec_t]

    # def _initDebugParaGUI(self):
    #     for key, wheel in self.wheelIds_dict.items():
    #         wheel["vecParaId"] = p.addUserDebugParameter(key + "_veclocity", -500, 500, wheel["velocity"])
    
    # def _changeGUIParas(self):
    #     for key, wheel in self.wheelIds_dict.items():
    #         targetVec = p.readUserDebugParameter(wheel["vecParaId"])
    #         wheel['velocity'] = targetVec


