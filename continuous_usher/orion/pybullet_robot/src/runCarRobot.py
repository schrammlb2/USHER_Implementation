
from CarRobot import CarRobot

if __name__ == "__main__":
    info_dict = {
        'jointId': {0: 'BackRight',
                    20: 'FrontRight',
                    40: 'BackLeft',
                    60: 'FrontLeft'},
        'init_motorStatus': {0: {'controlMode': 0, 'targetVelocity': 8, 'force': 5},
                             20: {'controlMode': 0, 'targetVelocity': 13, 'force': 5},
                             40: {'controlMode': 0, 'targetVelocity': 8, 'force': 5},
                             60: {'controlMode': 0, 'targetVelocity': 8, 'force': 5}},
        'init_physical_para': {0: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
                              20: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
                              40: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0},
                              60: {'lateralFriction': .7, 'angularDamping': 0, 'linearDamping': 0, 'rollingFriction': 0, 'spinningFriction': 0}}}

    options = {'global_wheel_control': False, 
                'GUI_friction': False, 
                'trajPlot': "show.png",
                'dat_savePath': './dat/t1.txt',
                'log_mp4': 'test.mp4'}

############# Test 1: racecar with initial mass #####################
    c = CarRobot("../mecanum_simple/mecanum_simple.urdf", info_dict, GUI=True,
                 options=options, timesteps = 600, debug=False, start_pos=[0, 0, 0.0007], start_ori = [0, 0, 0, 1])
    c.run()
