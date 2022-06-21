import numpy as np  #type: ignore
import pygame       #type: ignore
from constants import *

from solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map, create_map_1
from alt_red_light_environment import create_red_light_map
import pdb 

BLACK = (0, 0, 0)
WHITE = (250, 250, 250)
GREY  = (100, 100, 100)
LIGHT_GREY  = (175, 175, 175)

RED   = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE  = (50, 50, 200)

YELLOW = (200, 200, 50)

LIGHT_RED   = (250, 100, 100)
LIGHT_GREEN   = (100, 250, 100)

blockSize = 50


colors = { 
	# EMPTY: WHITE,	
    EMPTY: LIGHT_GREY,   		                       
	BLOCK: BLACK,                                   
	WIND:  WHITE,
	RANDOM_DOOR: GREY,
    # BREAKING_DOOR: LIGHT_RED,
    BREAKING_DOOR: RED,
    NONBREAKING_DOOR: LIGHT_GREEN,
}



def display_init(env) -> None:
# def display_init(grid: np.ndarray) -> None:
    grid: np.ndarray = env.grid

    global SCREEN, CLOCK
    global WINDOW_WIDTH
    global WINDOW_HEIGHT
    WINDOW_WIDTH = blockSize*grid.shape[0]
    WINDOW_HEIGHT = blockSize*grid.shape[1]
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    # while True:
    draw_grid(env)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()

def blitRotateCenter(surf, image, topleft, angle):

    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect)


def draw_grid(env, plot_agent = False, filename="_") -> None:
    grid: np.ndarray = env.grid
    min_val = 0
    max_val = 1

# def draw_grid(grid: np.ndarray) -> None:
    # for x in range(0, WINDOW_WIDTH, blockSize):
    #     for y in range(0, WINDOW_HEIGHT, blockSize):
    for i in range(0, grid.shape[0]):
        for j in range(0, grid.shape[1]):
            x = i*blockSize
            y = j*blockSize
            rect = pygame.Rect(x, y, blockSize, blockSize)
            block_type = grid[i,j]

            # if block_type != EMPTY: 
            #     color = colors[block_type]
            #     pygame.draw.rect(SCREEN, color, rect, 0)
            # else: 
            #     color = val_to_color(q.state_value(np.array([i,j])).max())
            #     pygame.draw.rect(SCREEN, color, rect, 0)
            if block_type == BLOCK: 
                color = colors[block_type]
                pygame.draw.rect(SCREEN, color, rect, 0)                
            else: 
                color = colors[block_type]
                pygame.draw.rect(SCREEN, color, rect, 0)
                if block_type == BREAKING_DOOR:
                    pt_list = [[x, y], [x+ blockSize, y], [x+ blockSize/2, y + blockSize]]
                    pygame.draw.polygon(SCREEN, YELLOW, pt_list)#, 2)   

                if block_type == NONBREAKING_DOOR:
                    pt_list = [[x, y], [x+ blockSize, y], [x+ blockSize/2, y + blockSize]]
                    pygame.draw.polygon(SCREEN, WHITE, pt_list)#, 2)
            # pygame.draw.rect(SCREEN, WHITE, rect, 1)
    scale = 1

    start = env.start
    start_rect = pygame.Rect(blockSize*(start[0]-scale/2), blockSize*(start[1]-scale/2), blockSize*scale, blockSize*scale)
    # pygame.draw.ellipse(SCREEN, WHITE, start_rect, 0)

    # goal = env.new_goal
    goal = env.goal
    goal_rect = pygame.Rect(blockSize*(goal[0]-scale/2), blockSize*(goal[1]-scale/2), blockSize*scale, blockSize*scale)
    pygame.draw.ellipse(SCREEN, BLACK, goal_rect, 0)

    if plot_agent:
        state = env.state
        pos = env.env.state_to_goal(state)
        x = pos[0]*blockSize
        y = pos[1]*blockSize
        w = env.width*blockSize
        l = env.length*blockSize
        rot = env.env.state_to_rot(state)
        signs = [1, -1]

        img = pygame.image.load(your_save_location_goes_here).convert_alpha()
        # img.convert()
        rect = img.get_rect()

        img = pygame.transform.scale(img, (l, w))
        # agent = pygame.Rect(x - w/2, y - l/2, w, l)
        # SCREEN.blit(img, agent)
        topleft = (x - l/2, y - w/2)
        # agent = pygame.Rect(x - w/2, y - l/2, w, l)
        # pygame.draw.rect(SCREEN, RED, agent, 0)
        if len(env.path) > 1:
            path = [(loc*blockSize).tolist() for loc in env.path]
            # import pdb
            # pdb.set_trace()
            pygame.draw.lines(SCREEN, RED, False, path)

        pi=3.1415962
        blitRotateCenter(SCREEN, img, topleft, (-rot)*180/pi)
        # pdb.set_trace()


    pygame.image.save(SCREEN, f"./logging/video/{filename}.png")

    pygame.display.update()

# main()
# display_init(np.ones((10, 10)))

# if __name__ == '__main__':
#     from solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map, create_map_1
#     from alt_red_light_environment import create_red_light_map

#     env = create_map_1(env_type="linear")
#     env.reset()
#     #write new function that includes the agent as a square

#     display_init(env)
#     #draw actor here
#     for i in range(10):
#         draw_grid(env, plot_agent = True, filename=f"linear_{i}")
#         env.step(np.array([1,0]))

if __name__ == '__main__':
    from solvable_gridworld_implementation import create_test_map, random_blocky_map, two_door_environment, random_map, create_map_1
    from alt_red_light_environment import create_red_light_map

    import os
    import re
    import pickle

    path = "logging/path_recordings/"
    env_name = "StandardCarGridworld"
    lst = sorted(os.listdir(path))

    for i in range(10):
        p= re.compile(f"{env_name}_{i}_*")
        matching_list = [p.match(string).string for string in lst if p.match(string) is not None]
        matching_list = sorted(matching_list, key=lambda x: int(x.split('_')[-1]))
        env = pickle.load(open(path + matching_list[0], "rb"))
        display_init(env)
        for j, file in enumerate(matching_list):
            env = pickle.load(open(path + file, "rb"))
            draw_grid(env, plot_agent = True, filename=file)
