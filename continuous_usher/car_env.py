import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from pyquaternion import Quaternion
import gym
from gym.spaces import Box
import pdb
# from utils import *
from orion.pybullet_robot.src.utils import *

from constants import *

arr = np.array
map_2D_to_4D_action = lambda action: arr([1,1,1,1])*action[0] + arr([-1,-1,1,1])*action[1]
UNBLOCKED_CHANCE = 0.5
length_extension = 0.0
BREAK = False

complete_block = False

FORWARD=arr([1,0])
RIGHT=arr([1,1])
LEFT=arr([1,-1])
STAY=arr([0,0])

class CarEnvironment(gym.core.GoalEnv):

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, vis=False) -> None:
        self.vis = vis
        # self.vis = False
        # friction_coefficients = {'BR': 0.536, 'FR': 0.597, 'BL': 0.805, 'FL': 1.039} #Initial guess
        friction_coefficients = {'BR': 0.536, 'BL': 0.597, 'FR': 0.805, 'FL': 1.039} #Edgar's instruction

        info_dict = {
            'jointId': {0: 'BackRight',
                    20: 'FrontRight',
                    40: 'BackLeft',
                    60: 'FrontLeft'},
            'init_motorStatus': {0: {'controlMode': 0, 'targetVelocity': 8, 'force': 5},
                             20: {'controlMode': 0, 'targetVelocity': 8, 'force': 5},
                             40: {'controlMode': 0, 'targetVelocity': 8, 'force': 5},
                             60: {'controlMode': 0, 'targetVelocity': 8, 'force': 5}},
            # 'init_physical_para': {0: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
            #                   20: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
            #                   40: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
            #                   60: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0}}}
            'init_physical_para': {0: {'lateralFriction':   friction_coefficients['BR'], 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
                              20: {'lateralFriction':       friction_coefficients['FR'], 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
                              40: {'lateralFriction':       friction_coefficients['BL'], 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
                              60: {'lateralFriction':       friction_coefficients['FL'], 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0}}}

        options = {'global_wheel_control': False, 
                'GUI_friction': False, 
                'trajPlot': "show.png",
                'dat_savePath': './dat/t1.txt',
                'log_mp4': 'test.mp4'}
        cwd = os.getcwd()
        robot_file = f"{cwd}/orion/pybullet_robot/mecanum_simple/mecanum_simple.urdf"
        timesteps=-1
        GUI=True
        debug=False
        options={}
        start_pos = [0, 0, 0.0007]
        start_ori =[0, 0, 0, 1]

        self.robot_file = robot_file
        self.info_dict = info_dict
        self.debug = debug
        self.GUI = vis
        self.ts = timesteps
        self.options = options

        self.info_dict['global_GUI_para'] = {}
        self.flag_exceed = True

        self.start_pos = start_pos
        self.start_ori = start_ori

        # self.goal = self.sample_goal()
        # self.action_space = Box(np.array([-1,-1,-1,-1]), np.array([1,1,1,1]))
        self.action_space = Box(np.array([-1,-1]), np.array([1,1]))
        self.time_per_agent_step = .5
        self.integration_steps_per_agent_step = 50
        self.pybullet_timestep = self.time_per_agent_step/self.integration_steps_per_agent_step

        self.reward_type = 'sparse'
        self.scale = .5
        self.distance_threshold = .5#self.scale
        self.setup()

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
        self.useRealTimeSim = False
        p.setRealTimeSimulation(self.useRealTimeSim)
        planeId = p.loadURDF("plane.urdf")
        # print("planeId: ", planeId)
        self.force_multiplier = 1
        p.setGravity(0,0,-10*self.force_multiplier)
        # p.setTimeStep(0.01)
        p.setTimeStep(self.pybullet_timestep)

    def _initializeMap(self, block):
        setup_dict = create_map(block)
        offset = setup_dict["start"]
        min_goal = setup_dict["goal_range"][0]
        max_goal = setup_dict["goal_range"][1]
        grid = setup_dict["grid"]
        # self.goal = min_goal + np.random.rand(2)*(max_goal-min_goal) - offset
        locations = [(i,j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
        ori = p.getQuaternionFromEuler([0, 0, 0])     
        self.blocks = []   
        block_height = 0#self.scale/2
        for loc in locations: 
            pos = ((arr(loc)-offset)*self.scale).tolist()
            # self.goal_vis = p.loadURDF('./urdf/block/brick_0.3/brick.urdf', pos, ori)
            if grid[loc] == BLOCK:
                # print(f"position: {pos}")
                self.blocks.append(p.loadURDF(f'./urdf/block/brick_{self.scale}/brick.urdf', pos + [block_height], ori))

        extra_block_loc = [3 + length_extension, 4]
        extra_block_pos = ((arr(extra_block_loc )-offset)*self.scale).tolist()
        self.blocks.append(p.loadURDF(f'./urdf/block/brick_{self.scale}/brick.urdf', extra_block_pos + [block_height], ori))
        if complete_block:
            for i in range(3, grid.shape[0]):
                extra_block_loc = [i, 4]
                extra_block_pos = ((arr(extra_block_loc )-offset)*self.scale).tolist()
                self.blocks.append(p.loadURDF(f'./urdf/block/brick_{self.scale}/brick.urdf', extra_block_pos + [block_height], ori))

        random_block_loc = [2,4]
        self.random_block_pos = ((arr(random_block_loc)-offset)*self.scale).tolist()
        # print(f"random_block position: {self.random_block_pos}")
        # self.random_block = p.loadURDF(f'./urdf/block/brick_{self.scale}/brick.urdf', self.random_block_pos + [block_height], ori)
        self.random_block = p.loadURDF(f'./urdf/block/brick_{self.scale}/brick.urdf', self.random_block_pos + [block_height], ori)
        self.blocked_config = p.saveState()

        lifted_pos = self.random_block_pos
        lifted_pos += [self.scale]
        p.resetBasePositionAndOrientation(self.random_block, lifted_pos, p.getQuaternionFromEuler([0,0,0]))
        self.unblocked_config = p.saveState()
        # pdb.set_trace()


    def _loadRobot(self, robot_file, start_pos, start_ori):
        self.robotId = p.loadURDF(robot_file, start_pos, start_ori, flags=p.URDF_USE_INERTIA_FROM_FILE)
        # print("robotId: ", self.robotId)

    def _initDynamicDumpings(self):
        p.changeDynamics(self.robotId, -1, linearDamping=0, angularDamping=0)
        for j in range(p.getNumJoints(self.robotId)):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)

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
        outval = np.concatenate([v for v in info.values()])
        
        return outval

    def _step(self):
        for jid, _ in self.info_dict['jointId'].items():
            targetVelocity = self.info_dict['init_motorStatus'][jid]['targetVelocity']
            maxForce = self.info_dict['init_motorStatus'][jid]['force']
            p.setJointMotorControl2(self.robotId,
                                    jid,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=targetVelocity,
                                    force=maxForce)#*self.force_multiplier)

        # if not self.useRealTimeSim:
        #     p.stepSimulation()
        #     time.sleep(0.01)
        for i in range(self.integration_steps_per_agent_step):
            p.stepSimulation()
            # contact_points = [p.getContactPoints(self.robotId, block) for block in self.blocks]
            contact_points = [p.getContactPoints(self.robotId, self.random_block)]
            if sum(contact_points, ()) != () and BREAK:
                self.broken = True
        info = self._get_info()
        self._stepCounter += 1
        return info
    
    def _dumpData(self):
        self.data = np.array(self.data)
        if 'dat_savePath' in self.options.keys():
            np.savetxt(self.options['dat_savePath'], self.data)
        if 'trajPlot' in self.options.keys():
            plotTrajectory(self.data, self.options['trajPlot'])

    def get_state(self):    	
        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        obs = np.concatenate([arr(pos + ori), self._get_info(), arr([self.broken*1])])
        ag  = arr(pos)
        state = {
        	"observation": obs,
        	"achieved_goal": ag[:2],
        	"desired_goal": self.goal
        }
        return state

    
    def is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        # print(d)
        return (d<self.distance_threshold)

    def step(self, action):
        if self.auto and self.i < len(action_sequence) - 1: 
            action = action_sequence[self.i]
            self.i += 1
        if self.broken: action *= 0
        max_vel = 20
        drivetrain_action = action
        drivetrain_action = map_2D_to_4D_action(action)
        # drivetrain_action = map_2D_to_4D_action(arr([1,0]))
        # drivetrain_action = arr([1,0])
        target_vel = max_vel*np.clip(drivetrain_action, -1, 1)
        for i in range(0, 4):
            self.info_dict['init_motorStatus'][i*20]['targetVelocity'] = target_vel[i]
        self._step()
        obs = self._get_info()
        state = self.get_state()
        reward = self.compute_reward(state['achieved_goal'], self.goal)
        done = reward == 0
        info = {"is_success": done}
        # print(state['achieved_goal'])
        return state, reward, done, info
        # pass

    def sample_goal(self):
        setup_dict = create_map()
        offset = setup_dict["start"]
        min_goal = setup_dict["goal_range"][0]
        max_goal = setup_dict["goal_range"][1]
        grid = setup_dict["grid"]
        goal = min_goal + np.random.rand(2)*(max_goal-min_goal) - offset
        # goal = (min_goal + 1/2*(max_goal-min_goal) - offset)
        goal *= self.scale
        return goal
    	# pass

    def goal_distance(self, goal_a, goal_b):
        goal_a = np.array(goal_a)
        goal_b = np.array(goal_b)
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    
    def compute_reward(self, achieved_goal, goal, info=None):
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def setup(self):        
        # p.restoreState(self.saved_state) 
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
        # base = p.saveState()
        self._initializeMap(block=False)
        # p.resetSimulation()


    def reset(self):
        self.broken = False
        self.i = 0
        # self.setup()
        if np.random.rand() < UNBLOCKED_CHANCE: 
            p.restoreState(self.unblocked_config)
            self.block_position = "unblocked"
        else: 
            p.restoreState(self.blocked_config)
            self.block_position = "blocked"
            # p.setPosition(self.random_block, lifted_pos)
        if np.random.rand() < 0.001: 
            self.auto = True
        else: 
            self.auto = False
        self.goal = self.sample_goal()
        self.initial_info = self._get_info()
        state = self.get_state()
        return state

    def close(self):
        p.disconnect(self.physicsClient)

def create_empty_map_grid(size):
    grid = np.zeros((size, size))
    mid = size//2
    for i in range(size):
        #Borders
        grid[0,i] = BLOCK
        grid[size-1,i] = BLOCK
        grid[i,0] = BLOCK
        grid[i, size-1] = BLOCK
    return grid

def create_map_grid(size, block):
    grid = create_empty_map_grid(size)
    mid = size//2
    #Wall through the middle
    long_block = 0
    for i in range(1, size//2 + long_block):
    # for i in range(1, size//2):
        grid[i,mid] = BLOCK
    if not block:
        grid[2,mid] = EMPTY
    return grid


def create_map(block=True):
    randomize_start = False
    size = 9
    grid = create_map_grid(size, block)
    mid = size//2
    offset = np.array([0, 0])
    start  = np.array([2,mid -2]) + offset
    # start  = np.array([2,mid -1]) + offset
    goal_center  = np.array([2, mid +2]) + offset
    diff = 1.
    goal_range = (goal_center-diff, goal_center+diff)

    return {
        "start": start, 
        "goal_range": goal_range,
        "grid": grid
    }


action_sequence = [RIGHT]*2 + [FORWARD]*4 + [LEFT]*2 + [FORWARD]*2 + [LEFT]*2 + [FORWARD]*4
# action_sequence = [RIGHT] + [(RIGHT+FORWARD)/2] + [FORWARD] + [LEFT*.75 + FORWARD*.25] + [LEFT*.25 + FORWARD*.75] + [LEFT*.75 + FORWARD*.25]
# action_sequence = sum([[i,i] for i in action_sequence], [])
print(len(action_sequence))

if __name__ == "__main__":
    env = CarEnvironment(vis=True)
    movement = "forward"
    if movement == "forward":
        BackRight=1
        FrontRight=1
        BackLeft=1
        FrontLeft=1
    elif movement == "diagonal":
        BackRight=1
        FrontRight=-1
        BackLeft=-1
        FrontLeft=1

    forward_action_sequence = [np.array([1,0]) for _ in range(20)]
    for i in range(10):
        env.reset()
        # for action in [a*.75 for a in action_sequence]: 
        # for action in action_sequence: 
        pdb.set_trace()
        for action in forward_action_sequence: 
            env.step(action)
            contact_points = [p.getContactPoints(env.robotId, block) for block in env.blocks]
            # print(action)
        for _ in range(6):
            env.step(np.array([0, 0]))
        # for i in range(6):
        # 	# c.step(np.array([BackRight,FrontRight,BackLeft,FrontLeft]))
        #     c.step(np.array([1, 0]))