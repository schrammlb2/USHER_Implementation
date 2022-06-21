import gym
from throwing_env import make_throwing_env
from car_env import CarEnvironment
from gym.wrappers.time_limit import TimeLimit
from action_randomness_wrapper import ActionRandomnessWrapper
from torus_env import Torus

def make_env(args):
    # create the ddpg_agent
    if args.env_name == "Throwing":
        env = TimeLimit(make_throwing_env(), max_episode_steps=20)
    elif "CarEnvironment" in args.env_name:
        env = TimeLimit(CarEnvironment(), max_episode_steps=50)
    elif "Gridworld" in args.env_name: 
        #Note: This is not actually a gridworld environment. It's just called that because the maps are made of square blocks
        # And also because I started with the discrete USHER implementation and then made it continuous
        if "Omnibot" in args.env_name:
            from omnibot_gridworld import create_map_1
        elif "RedLight" in args.env_name: 
            from red_light_environment import create_red_light_map as create_map_1
        else:
            from navigation_environment import create_map_1#create_test_map, random_blocky_map, two_door_environment, random_map, create_map_1


        max_steps = 50 if "Alt" or "RedLight" in args.env_name in args.env_name else 20
        if args.env_name == "TwoDoorGridworld":
            env=TimeLimit(two_door_environment(), max_episode_steps=50)
        else:
            mapmaker = create_map_1

            if "Asteroids" in args.env_name: 
                env_type="asteroids"
            elif "StandardCar" in args.env_name:
                env_type = "standard_car"
            elif "Car" in args.env_name:
                env_type = "car"
            elif "Omnibot" in args.env_name:
                env_type = "omnibot"
            else: 
                env_type = "linear"
            print(f"env type: {env_type}")
            env = TimeLimit(mapmaker(env_type=env_type), max_episode_steps=max_steps)
    elif "Torus" in args.env_name:
        freeze = "Freeze" in args.env_name or "freeze" in args.env_name
        if freeze: 
            n = args.env_name[len("TorusFreeze"):]
        else: 
            n = args.env_name[len("Torus"):]
        try: 
            dimension = int(n)
        except:
            print("Could not parse dimension. Using n=2")
            dimension=2
        print(f"Dimension = {dimension}")
        print(f"Freeze = {freeze}")
        env = TimeLimit(Torus(dimension, freeze), max_episode_steps=50)
    else:
        env = gym.make(args.env_name)

    env = ActionRandomnessWrapper(env, args.action_noise)

    return env