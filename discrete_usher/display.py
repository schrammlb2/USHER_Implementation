import numpy as np  #type: ignore
import pygame       #type: ignore
from constants import *

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
GREY  = (100, 100, 100)

RED   = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE  = (50, 50, 200)

YELLOW = (200, 200, 50)

LIGHT_RED   = (250, 100, 100)
LIGHT_GREEN   = (100, 250, 100)

blockSize = 50


colors = { 
	EMPTY: WHITE,			                       
	BLOCK: BLACK,                                   
	WIND:  WHITE,
	RANDOM_DOOR: GREY,
    # BREAKING_DOOR: LIGHT_RED,
    BREAKING_DOOR: RED,
    NONBREAKING_DOOR: LIGHT_GREEN,
}



def display_init(env, q: np.ndarray) -> None:
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
    draw_grid(env, q)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()


def draw_grid(env, q: np.ndarray) -> None:
    grid: np.ndarray = env.grid
    min_val = 0
    max_val = 1
    val_to_color = lambda q: tuple(q*np.array(GREEN) + (1-q)*np.array(RED))
    # val_to_color = lambda q: WHITE

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
                color = val_to_color(q.state_value(np.array([i,j]), policy="Q").max())
                # color = val_to_color(q.state_value(np.array([i,j]), policy="HER").max())
                pygame.draw.rect(SCREEN, color, rect, 0)
                if block_type == BREAKING_DOOR:
                    pt_list = [[x, y], [x+ blockSize, y], [x+ blockSize/2, y + blockSize]]
                    pygame.draw.polygon(SCREEN, YELLOW, pt_list)#, 2)   

                if block_type == NONBREAKING_DOOR:
                    pt_list = [[x, y], [x+ blockSize, y], [x+ blockSize/2, y + blockSize]]
                    pygame.draw.polygon(SCREEN, WHITE, pt_list)#, 2)   
            pygame.draw.rect(SCREEN, WHITE, rect, 1)


    start = env.start
    start_rect = pygame.Rect(blockSize*start[0], blockSize*start[1], blockSize, blockSize)
    color = val_to_color(q.state_value(np.array([start[0],start[1]])).max())
    pygame.draw.rect(SCREEN, color, start_rect, 0)
    pygame.draw.ellipse(SCREEN, WHITE, start_rect, 0)

    goal = env.new_goal
    goal_rect = pygame.Rect(blockSize*goal[0], blockSize*goal[1], blockSize, blockSize)
    pygame.draw.ellipse(SCREEN, BLACK, goal_rect, 0)


    pygame.display.update()

    pygame.image.save(SCREEN, f"./figures/Q_value_map.png")

# main()
# display_init(np.ones((10, 10)))