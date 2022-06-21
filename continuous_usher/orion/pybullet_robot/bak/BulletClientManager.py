import pybullet_utils.bullet_client as bc
import pybullet
import pybullet_data
import os
import yaml

from pybullet_robot.CarRobot import CarRobot
#from NN_model.data_utils.BulletObservations import SimpleObs

## https://github.com/bulletphysics/bullet3/issues/1925

class BulletClientManager():
    """
    PyBullet ClientManager:
    Generate multiple trajectories by opening n amount of bullet client
    @ num_client: Number of trajectory / client
    @ ObsManager: Specify Observation Sets
    @ output_dir: output .pkl files directory
    @ debug: True if wanna print out observation infos
    """
    def __init__(self, num_client, ObsManager, output_dir = None, debug = False):
        self.num_client = num_client
        self.client_dict = {}
        self.ObsManager = ObsManager
        self.debug = debug
        self.output_dir = output_dir
        for i in range(self.num_client):
            if self.num_client == 1:
                p = bc.BulletClient(connection_mode=pybullet.GUI)
                self.GUI = True
            else:
                p = bc.BulletClient(connection_mode=pybullet.DIRECT)
                self.GUI = False
            self.client_dict[p] = {}
        
        self.counter = 0
    
    # def _reset(self):
    #     for p, client in self.client_dict.items():
    #         self._clientInitilaize(p)
    
    def _clientInitilaize(self, p):
        """
        Add basic Plane to each client
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setTimeStep(0.01)
        p.loadURDF("plane.urdf")
        p.setGravity(0,0,-10)
        
    def _initCarRobot(self, p):
        setting = self.client_dict[p]['setting']
        c = CarRobot(clientP = p, ObsManager = self.ObsManager, setting=setting)
        self.client_dict[p]['CarRobot'] = c

    def setTraj(self, yamls_files):
        """
        Set up yamls to generate trajectories
        Must be called before run()
        """
        for i in range(self.num_client):
            p = list(self.client_dict)[i]
            yaml_name = yamls_files[i]
            with open(yaml_name, 'rb') as f:
                setting = yaml.load(f, Loader=yaml.Loader)
            
            ## Load EXTRA (not written in yamls) setting from function call
            ## setting: the settings that will parsed to class CarRobot()
            setting['output_dir'] = self.output_dir ## TODO
            setting['debug'] = self.debug
            setting['GUI'] = self.GUI
            self.client_dict[p]['setting'] = setting

    def run(self):
        print("Starting Generate Trajectory pkl files ......")
        for p, _ in self.client_dict.items():
            ## Initialize and load default plane and robot
            self._clientInitilaize(p)
            self._initCarRobot(p)
        
        ## Tried multiprocessing but not working
        for _, client in self.client_dict.items():
            self.counter += 1
            print('{} / {}: Processing file {}.pkl'.format(self.counter, self.num_client, client['setting']['basename']),end='\r')
            car_robot = client['CarRobot']
            car_robot.run()
        print()