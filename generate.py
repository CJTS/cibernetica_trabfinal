import sys
import Game
import ff
import numpy as np
import ctypes
import copy
from helper import get_state, preprocess_data

data_per_level = 200  # Unique samples per level

def find_plan(game, subgoal):
    """Find a plan from the current state to the given subgoal."""
    plan = []
    with open("problem.pddl", "w") as f:
        f.write(game.get_problem(subgoal))
    plan_result = ff.plan((ctypes.c_char_p * 7)(b"ff", b"-f", b"./problem.pddl", b"-o", b"./domain.pddl", b"-i", b"0"))
    plan = []
    i = 0
    attainable = False
    if plan_result:
        while plan_result[i]:
            attainable = True
            plan.append(plan_result[i].decode('utf-8'))  # Decode the C string to Python string
            i += 1
    ff.free_memory(plan_result)
    plan.reverse()

    return plan, attainable

def execute_plan(game, plan):
    """Execute a plan"""
    action = plan.pop()
    while action:
        if action == "MOVE-UP":
            game._move(Game.Direction.UP)
        elif action == "MOVE-DOWN":
            game._move(Game.Direction.DOWN)
        elif action == "MOVE-RIGHT":
            game._move(Game.Direction.RIGHT)
        elif action == "MOVE-LEFT":
            game._move(Game.Direction.LEFT)
        elif action == "USE-UP":
            game._use()
        elif action == "USE-DOWN":
            game._use()
        elif action == "USE-LEFT":
            game._use()
        elif action == "USE-RIGHT":
            game._use()
        game.play_step()
        if(len(plan) > 0):
            action = plan.pop()
        else:
            break

def collect_samples(game):
    """Collect samples for training using random exploration."""
    dataset = []
    print("Level: ", sys.argv[1])
    level_samples = 0
    game.init_game()
    state = get_state(game)

    # Copy init state to backtrack until all data per level
    init_player = game.player
    init_grid = [[0 for _ in range(Game.GRID_WIDTH)] for _ in range(Game.GRID_HEIGHT)]
    for x in range(len(init_grid)):
        for y in range(len(init_grid[x])):
            init_grid[x][y] = Game.Tiles(game.grid[x][y])

    while level_samples < data_per_level:
        while True:
            subgoal = game.choose_subgoal()

            if subgoal != None:
                plan, attainable = find_plan(game, subgoal)

                if attainable:
                    plan_length = len(plan)
                    execute_plan(game, plan)
                    sample = (state, game.gems[subgoal], plan_length)
                    dataset.append(sample)
                    level_samples += 1
                    print("Data in level: ", level_samples)
                else:
                    sample = (state, subgoal, 0)
                    dataset.append(sample)
                    level_samples += 1
                    print("Data in level: ", level_samples)

                if level_samples >= data_per_level:
                    break
            else:
                print("Finished all subgoals level, starting again.")
                game.reset_game(copy.deepcopy(init_grid), init_player)
                break

    states, targets = preprocess_data(dataset)
    np.save('dataset2/states-' + str(sys.argv[1]) + '.npy', np.array(states))
    np.save('dataset2/targets-' + str(sys.argv[1]) + '.npy', np.array(targets))

if __name__ == "__main__":
    game = Game.BoulderDash()
    print("Collecting samples...")
    collect_samples(game)
