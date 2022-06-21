import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from pyquaternion import Quaternion

# from utils import *
from orion.pybullet_robot.src.utils import *
# from plotWheelForces import *

class CarRobot():
    def __init__(self, robot_file, info_dict, timesteps=-1, GUI=False, debug=False, options={}, 
                 start_pos = [0, 0, 0.0007], start_ori =[0, 0, 0, 1]):
        self.robot_file = robot_file
        self.info_dict = info_dict
        self.debug = debug
        self.GUI = GUI
        self.ts = timesteps
        self.options = options

        self.info_dict['global_GUI_para'] = {}
        self.flag_exceed = True

        self.start_pos = start_pos
        self.start_ori = start_ori
    
    def _initializeDumping(self):
        self.data = []

    def _initialize(self):
        self._stepCounter = 0
        self._initializeDumping()

        if self.GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setTimeStep(0.01)
        self.useRealTimeSim = False
        p.setRealTimeSimulation(self.useRealTimeSim)
        planeId = p.loadURDF("plane.urdf")
        print("planeId: ", planeId)

    def _loadRobot(self, robot_file, start_pos, start_ori):
        self.robotId = p.loadURDF(robot_file,
                                  start_pos, start_ori, flags=p.URDF_USE_INERTIA_FROM_FILE)
        print("robotId: ", self.robotId)

    def _initDynamicDumpings(self):
        p.changeDynamics(self.robotId, -1, linearDamping=0, angularDamping=0)
        for j in range(p.getNumJoints(self.robotId)):
            p.changeDynamics(self.robotId, j, linearDamping=0,
                             angularDamping=0)

    def _initDebugParaGUI(self):
        if not self.GUI:
            return

        if self.options == {}:
            return
        else:
            self.info_dict['joint_GUI_para'] = {}
            for jid, _ in self.info_dict['jointId'].items():
                self.info_dict['joint_GUI_para'][jid] = {}

        if self.options['global_wheel_control']:
            vec_default_ref = self.info_dict['init_motorStatus'][2]['targetVelocity']
            force_default_ref = self.info_dict['init_motorStatus'][2]['force']
            self.info_dict['global_GUI_para']['targetVelocitySlider'] = p.addUserDebugParameter(
                "global_targetVelocity", -30, 30, vec_default_ref)
            self.info_dict['global_GUI_para']['maxForceSlider'] = p.addUserDebugParameter(
                "global_force", 0, 10, force_default_ref)
        else:
            for jid, para in self.info_dict['joint_GUI_para'].items():
                para["vecId"] = p.addUserDebugParameter(
                    self.info_dict['jointId'][jid] + "_targetVelocity", -30, 30, self.info_dict['init_motorStatus'][jid]['targetVelocity'])  # self.info_dict['init_motorStatus'][jid]['targetVelocity']
            for jid, para in self.info_dict['joint_GUI_para'].items():
                para["forceId"] = p.addUserDebugParameter(
                    self.info_dict['jointId'][jid] + "_force", 0, 10, self.info_dict['init_motorStatus'][jid]['force'])  # self.info_dict['init_motorStatus'][jid]['force']

        if self.options['GUI_friction']:
            for jid, para in self.info_dict['joint_GUI_para'].items():
                para["lateral_fricId"] = p.addUserDebugParameter(
                    self.info_dict['jointId'][jid] + "_fric", 0, 100, 0.5)

    def _initMotorStatus(self):
        if self.info_dict['init_motorStatus'] == {}:
            return
        for jid, para in self.info_dict['init_motorStatus'].items():
            para['bodyUniqueId'] = self.robotId
            para['jointIndex'] = jid
            p.setJointMotorControl2(**para)

    def _initPhysicalParas(self):
        if self.info_dict['init_physical_para'] == {}:
            return

        for jid, para in self.info_dict['init_physical_para'].items():
            para['bodyUniqueId'] = self.robotId
            para['linkIndex'] = jid
            p.changeDynamics(**para)

    def _get_info(self):
        """
        Info will be printed out in each frame
        Data will be saved to a pickle file
        """
        linearVec, angularVec = p.getBaseVelocity(self.robotId)
        cps = p.getContactPoints(bodyA = 0) ## planeId
        
        intergration = True
        
        if intergration:
            info = {"FL": np.zeros(3), "BL": np.zeros(3), "FR": np.zeros(3), "BR": np.zeros(3)}
            key = ["part", "lateralFriction"]
            for cp in cps:
                f = getLateralFrictionV3(cp[10], cp[11], cp[12], cp[13], tolerance = 1e-06)
                if np.allclose(f, np.zeros(3)):
                    continue
                w = getWheel(cp[4])
                info[w] += f
                #plotFOnCar(info,linearVec)
        else:  ## Show different contact point parts
            info = {}
            cps = p.getContactPoints(bodyA = 0)
            count = 1
            for cp in cps:
                f = getLateralFrictionV3(cp[10], cp[11], cp[12], cp[13], tolerance = 1e-06)
                if np.allclose(f, np.zeros(3)):
                    continue
                component = getPart(cp[4])
                # info[count] = {}
                info[count]=dict(zip([component], [f.tolist()]))
                count += 1

        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        xytheta = GetXYTheta(pos, ori)
        self.data.append(xytheta)
        
        return info

    def _step(self):
        if self.GUI:
            if self.options != {}:
                if self.options['global_wheel_control']:
                    targetVelocity = p.readUserDebugParameter(
                        self.info_dict['global_GUI_para']['targetVelocitySlider'])
                    maxForce = p.readUserDebugParameter(
                        self.info_dict['global_GUI_para']['maxForceSlider'])
                for jid, _ in self.info_dict['jointId'].items():
                    wheel = self.info_dict['joint_GUI_para'][jid]
                    if not self.options['global_wheel_control']:
                        targetVelocity = p.readUserDebugParameter(
                            wheel["vecId"])
                        maxForce = p.readUserDebugParameter(wheel["forceId"])
                    p.setJointMotorControl2(self.robotId,
                                            jid,
                                            p.VELOCITY_CONTROL,
                                            targetVelocity=targetVelocity,
                                            force=maxForce)
                    if self.options['GUI_friction']:
                        tmp = p.readUserDebugParameter(wheel["lateral_fricId"])
                        p.changeDynamics(bodyUniqueId=self.robotId,
                                         linkIndex=jid, lateralFriction=tmp)

        if not self.useRealTimeSim:
            p.stepSimulation()
            time.sleep(0.01)
        info = self._get_info()
        self._stepCounter += 1
        return info

    def _Main_Loop(self):
        import pprint
        if self.ts > 1:
            for i in range(self.ts):
                info = self._step()
                if self.debug:
                    # print(info, end='\r')
                    pprint.pprint(info)
        else:
            while(1):
                info = self._step()
                if self.debug:
                    # print(info, end='\r')
                    pprint.pprint(info)
    
    def _dumpData(self):
        self.data = np.array(self.data)
        if 'dat_savePath' in self.options.keys():
            np.savetxt(self.options['dat_savePath'], self.data)
        if 'trajPlot' in self.options.keys():
            plotTrajectory(self.data, self.options['trajPlot'])

    def run(self):

        self._initialize()
        self._loadRobot(self.robot_file, self.start_pos, self.start_ori)
        try:
            if self.options['log_mp4'] is not None:
                self.logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.options['log_mp4'], [self.robotId])
        except:
            pass
        self._initDebugParaGUI()

        self._initDynamicDumpings()
        self._initMotorStatus()
        self._initPhysicalParas()

        # for j in range(p.getNumJoints(self.robotId)):
        #     info = p.getJointInfo(self.robotId, j)
        #     print(j, info)

        self._Main_Loop()
        self._dumpData()
        try:
            p.stopStateLogging(self.logId)
        except:
            pass

        p.disconnect()


# if __name__ == "__main__":
#     info_dict = {
#         'jointId': {2: 'left_rear',
#                     3: 'right_rear',
#                     5: 'left_front',
#                     7: 'right_front'},
#         'init_motorStatus': {2: {'controlMode': 0, 'targetVelocity': 10.57, 'force': 5},
#                              3: {'controlMode': 0, 'targetVelocity': 10.57, 'force': 5},
#                              5: {'controlMode': 0, 'targetVelocity': 10.57, 'force': 5},
#                              7: {'controlMode': 0, 'targetVelocity': 10.57, 'force': 5}},
#         'init_physical_para': {2: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0},
#                                3: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0},
#                                5: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0},
#                                7: {'angularDamping': 0, 'linearDamping': 0, 'lateralFriction': .5, 'rollingFriction': 0, 'spinningFriction': 0}}}

#     options = {'global_wheel_control': True, 'GUI_friction': False}

# ############# Test 5: (Modify Mass inside bullet) racecar with 100 * initial mass #####################
# ## 1 (400.0, 0.5, (0.0002666666666666667, 0.0002666666666666667, 0.0002666666666666667)
#     # ##TypeError: 'localInertiaDiagnoal' is an invalid keyword argument for this function
#     c = CarRobot("../racecar_Phy/racecar/racecar_4000.urdf", info_dict, GUI=True,
#                  options=options, debug=True, start_pos=[0, 0, 0.0007], start_ori = [0, 0, 0.70710678, 0.70710678])
#     c.run()
