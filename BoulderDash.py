import random
import time
import ctypes
import pygame
import random
from enum import Enum
from collections import namedtuple
import ff

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Tiles(Enum):
    PLAYER = 1
    EXIT = 2
    BOULDER = 3
    GEM = 4
    WALL = 5
    DIRT = 6
    SUBGOAL = 7
    EMPTY = 8

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GEM = (237, 48, 207)
DIRT = (103, 60, 14)
BOULDER = (158, 155, 149)

BLOCK_SIZE = 20
DIRT_NUM = 800
GEMS_NUM = 23
BOULDER_NUM = 100
SPEED = 20
GRID_SIZE = 30

class BoulderDash:
    def __init__(self, with_ui = False, w = GRID_SIZE * BLOCK_SIZE, h = GRID_SIZE * BLOCK_SIZE):
        
        self.w = w
        self.h = h
        # init display
        self.with_ui = with_ui
        if self.with_ui:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Boulderdash')

    def init_game(self, with_ai = False):
        if self.with_ui:
            self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT

        self.score = 0
        self.with_ai = with_ai
        self.player = None
        self.collecting = None
        self.collected = []
        w, h = GRID_SIZE, GRID_SIZE
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                self.grid[x][y] = Tiles.EMPTY

        self._place_dirt()
        self._place_boulders()
        self._place_gem()
        self._place_player()
        self.plan = []

    def _place_player(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
        while((x,y) in self.gems):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
        self.grid[x][y] = Tiles.PLAYER
        self.player = Point(x, y)

    def _place_dirt(self):
        for _ in range(DIRT_NUM):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
            self.grid[x][y] = Tiles.DIRT

    def _place_gem(self):
        self.gems = []
        for _ in range(GEMS_NUM):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
            self.gems.append((x,y))
            self.grid[x][y] = Tiles.GEM

    def _place_boulders(self):
        for _ in range(BOULDER_NUM):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
            self.grid[x][y] = Tiles.BOULDER

    def play_step(self):
        game_over = False

        # 1. collect user input
        if self.with_ui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self._move(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self._move(Direction.RIGHT)
                    elif event.key == pygame.K_UP:
                        self._move(Direction.UP)
                    elif event.key == pygame.K_DOWN:
                        self._move(Direction.DOWN)
                    elif event.key == pygame.K_RETURN:
                        self._use()
                    elif event.key == pygame.K_ESCAPE:
                        game_over = True

        # AI Planner
        if self.with_ai:
            if(len(self.plan) > 0):
                action = self.plan.pop()
                print("Action: ", action)
                if action == "MOVE-UP":
                    self._move(Direction.UP)
                elif action == "MOVE-DOWN":
                    self._move(Direction.DOWN)
                elif action == "MOVE-RIGHT":
                    self._move(Direction.RIGHT)
                elif action == "MOVE-LEFT":
                    self._move(Direction.LEFT)
                elif action == "USE-UP":
                    self._use()
                elif action == "USE-DOWN":
                    self._use()
                elif action == "USE-LEFT":
                    self._use()
                elif action == "USE-RIGHT":
                    self._use()
            else:
                self.plan = []
                with open("problem.pddl", "w") as f:
                    diamondRandomIndex = self.choosed_subgoal()
                    print('Going to ', diamondRandomIndex)
                    f.write(self.get_problem(diamondRandomIndex))

                print("Planning")
                plan_result = ff.plan((ctypes.c_char_p * 7)(b"ff", b"-f", b"./problem.pddl", b"-o", b"./domain.pddl", b"-i", b"0"))
                print("Finished Planning")

                i = 0
                print("Plan: ", plan_result)
                while plan_result[i]:
                    self.plan.append(plan_result[i].decode('utf-8'))  # Decode the C string to Python string
                    i += 1
                self.plan.reverse()
                # print(self.plan)

        # 3. check collisions
        # self._check_collisions()

        # 4. check if game over
        if self.score >= 5:
            game_over = True
            return game_over, self.score

        # 5. update ui and clock
        if self.with_ui:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. return game over and score

        # time.sleep(.5)
        return game_over, self.score

    def _check_collisions(self,x ,y):
        tile = self.grid[x][y]
        if tile == Tiles.DIRT:
            self.grid[x][y] = Tiles.EMPTY
        elif tile == Tiles.GEM:
            self.grid[x][y] = Tiles.EMPTY
            self.collected.append((x,y))
            self.score += 1

    def _is_collisions(self, newPlayer):
        if newPlayer.x > GRID_SIZE - 1 or newPlayer.x < 0 or newPlayer.y > GRID_SIZE - 1 or newPlayer.y < 0:
            return True

        tile = self.grid[newPlayer.x][newPlayer.y]
        if tile == Tiles.BOULDER:
            return True

    def can_push_boulder(self, newPlayer, direction):
        if direction == Direction.RIGHT and self.grid[newPlayer.x][newPlayer.y + 1] == Tiles.EMPTY:
            return True
        elif direction == Direction.LEFT and self.grid[newPlayer.x][newPlayer.y - 1] == Tiles.EMPTY:
            return True
        return False

    def push_boulder(self, newPlayer, direction):
        if direction == Direction.RIGHT:
            self.grid[newPlayer.x][newPlayer.y + 1] = Tiles.BOULDER
            self.grid[newPlayer.x][newPlayer.y] = Tiles.EMPTY
        elif direction == Direction.LEFT:
            self.grid[newPlayer.x][newPlayer.y - 1] = Tiles.BOULDER
            self.grid[newPlayer.x][newPlayer.y] = Tiles.EMPTY

    def _move(self, direction):
        if self.direction == direction:
            x = self.player.x
            y = self.player.y
            if direction == Direction.RIGHT:
                y += 1
            elif direction == Direction.LEFT:
                y -= 1
            elif direction == Direction.DOWN:
                x += 1
            elif direction == Direction.UP:
                x -= 1

            newPlayer = Point(x, y)
            if(self._is_collisions(newPlayer)):
                if(self.can_push_boulder(newPlayer, direction)):
                    self.push_boulder(newPlayer, direction)
                else:
                    return

            self.player = newPlayer
            self._check_collisions(x, y)
            self.grid[x][y] = Tiles.PLAYER
        else:
            self.direction = direction

    def _use(self):
        if(self.direction == Direction.UP and self.grid[self.player.x - 1][self.player.y] == Tiles.BOULDER):
            self.grid[self.player.x - 1][self.player.y] = Tiles.EMPTY
        elif(self.direction == Direction.DOWN and self.grid[self.player.x + 1][self.player.y] == Tiles.BOULDER):
            self.grid[self.player.x + 1][self.player.y] = Tiles.EMPTY
        elif(self.direction == Direction.LEFT and self.grid[self.player.x][self.player.y - 1] == Tiles.BOULDER):
            self.grid[self.player.x][self.player.y - 1] = Tiles.EMPTY
        elif(self.direction == Direction.RIGHT and self.grid[self.player.x][self.player.y + 1] == Tiles.BOULDER):
            self.grid[self.player.x][self.player.y + 1] = Tiles.EMPTY

    def _update_ui(self):
        gemindex = 0
        if self.with_ui:
            self.display.fill(BLACK)

            for x in range(len(self.grid)):
                for y in range(len(self.grid[x])):
                    if self.grid[x][y] == Tiles.DIRT:
                        pygame.draw.rect(self.display, DIRT, pygame.Rect(y * BLOCK_SIZE, x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    elif self.grid[x][y] == Tiles.BOULDER:
                        pygame.draw.rect(self.display, BOULDER, pygame.Rect(y * BLOCK_SIZE, x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                    elif self.grid[x][y] == Tiles.GEM:
                        pygame.draw.rect(self.display, GEM, pygame.Rect(y * BLOCK_SIZE, x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                        gemindex += 1

            pygame.draw.rect(self.display, RED, pygame.Rect(self.player.y  * BLOCK_SIZE, self.player.x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

            font = pygame.font.Font('arial.ttf', 25)
            text = font.render("Score: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])

            for x in range(len(self.gems)):
                font = pygame.font.Font('arial.ttf', 16)
                text = font.render(str(x), True, WHITE)
                self.display.blit(text, [self.gems[x][1] * BLOCK_SIZE, self.gems[x][0] * BLOCK_SIZE])

            pygame.display.flip()

    def count_diamonds(self):
        index = 0
        diamonds = []
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == Tiles.GEM:
                    diamonds.append(index)
                    index += 1
        return index

    def get_gems(self):
        pddl = "    "
        for x in range(len(self.gems)):
            pddl += "gem" + str(x) + " "
        pddl += "- gem\n"
        return pddl

    def get_boulders(self):
        pddl = "    "
        index = 1
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == Tiles.BOULDER:
                    pddl += "r" + str(index) + " "
                    index += 1
        pddl += "- boulder\n"
        return pddl if index > 1 else ""

    def get_cells(self):
        index = 1
        pddl = "    "
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                pddl += "c_" + str(x) + "_" + str(y) + " "
                index += 1
        pddl += "- cell\n"
        return pddl

    def get_state(self):
        indexBoulder = 1
        indexDirt = 1
        indexGems = 0
        pddl = "    "
        pddl += "(at p1 c_" + str(self.player.x) + "_" + str(self.player.y) +")\n    "
        if self.direction == Direction.UP:
            pddl += "(oriented-up p1)\n    "
        elif self.direction == Direction.DOWN:
            pddl += "(oriented-down p1)\n    "
        elif self.direction == Direction.LEFT:
            pddl += "(oriented-left p1)\n    "
        elif self.direction == Direction.RIGHT:
            pddl += "(oriented-right p1)\n    "
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == Tiles.BOULDER:
                    pddl += "(at r" + str(indexBoulder) + " c_" + str(x) + "_" + str(y) + ")\n    "
                    indexBoulder += 1
                # elif self.grid[x][y] == Tiles.GEM:
                #     pddl += "(at gem" + str(indexGems) + " c_" + str(x) + "_" + str(y) + ")\n    "
                #     indexGems += 1
                # elif self.grid[x][y] == Tiles.DIRT:
                #     pddl += "(at d" + str(indexDirt) + " c_" + str(x) + "_" + str(y) + ")\n    "
                #     indexDirt += 1
                elif self.grid[x][y] == Tiles.EMPTY:
                    pddl += "(terrain-empty c_" + str(x) + "_" + str(y) + ")\n    "

        for x in range(len(self.gems)):
            pddl += "(at gem" + str(x) + " c_" + str(self.gems[x][0]) + "_" + str(self.gems[x][1]) + ")\n    "

        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if(x > 0):
                    pddl += "(connected-up c_" + str(x) + "_" + str(y) + " c_" + str(x - 1) + "_" + str(y) + ")\n    "
                if(x < GRID_SIZE - 1):
                    pddl += "(connected-down c_" + str(x) + "_" + str(y) + " c_" + str(x + 1) + "_" + str(y) + ")\n    "
                if(y > 0):
                    pddl += "(connected-left c_" + str(x) + "_" + str(y) + " c_" + str(x) + "_" + str(y - 1) + ")\n    "
                if(y < GRID_SIZE - 1):
                    pddl += "(connected-right c_" + str(x) + "_" + str(y) + " c_" + str(x) + "_" + str(y + 1) + ")\n    "
        return pddl

    def get_problem(self, dIndex = 1):
        pddl = """(define (problem boulder_dash_problem)
  (:domain boulder_dash)

  ;; Objects
  (:objects
    p1 - player
"""
        pddl += self.get_gems()
        pddl += self.get_boulders()
        pddl += self.get_cells()
        pddl += """  )\n
  ;; Initial State
  (:init\n"""
        pddl += self.get_state()
        pddl += ")\n\n"
        pddl += """  ;; Goal State
  (:goal
    (and
    """
        pddl += "  (got gem" + str(dIndex) + ")"
        pddl += """
     )
   )
 )"""

        return pddl

    def subgoals(self):
        return [item for item in self.gems if item not in self.collected]

    def choosed_subgoal(self):
        remainingGems = self.subgoals()
        if(len(remainingGems) > 0):
            remainingGemsIndex = random.randrange(0, len(remainingGems))
            diamondRandomIndex = self.gems.index(remainingGems[remainingGemsIndex])
            return diamondRandomIndex
        else:
            return None

    def reset_game(self, init_grid, init_player):
        self.grid = init_grid
        self.player = init_player
        self.collected = []
        self.direction = Direction.RIGHT
        self.score = 0
        self.collecting = None
        self.collected = []
        self._update_ui()

if __name__ == '__main__':
    game = BoulderDash(True)
    game.init_game(True)

    # game loop
    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break
    # game_over, score = game.play_step()
    # print("planning")
    # with open("problem.pddl", "w") as f:
    #     f.write(game.get_problem())
    # plan_result = ff.plan((ctypes.c_char_p * 7)(b"ff", b"-f", b"./problem.pddl", b"-o", b"./domain.pddl", b"-i", b"0"))
    # print("plan result", plan_result)
    # if plan_result:
    #     for action in plan_result:
    #         if(action == None):
    #             break
    #         print(action)


    print('Final Score', score)

    pygame.quit()
