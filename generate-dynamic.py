import sys
import Game
import ff
import numpy as np
import ctypes
import copy
from Helper import get_state, preprocess_data, input_shape

data_per_level = 500  # Unique samples per level
final_reward = -10
penalization = 10

def find_plan(game, subgoal):
    """Find a plan from the current state to the given subgoal."""
    plan = []
    with open("problem.pddl", "w") as f:
        if(subgoal == None):
            f.write(game.get_problem_exit())
        else:
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
        print("Data in level:", level_samples)
        subgoals = game.subgoals()
        subgoal = game.choose_subgoal() # Escolha o subobjetivo
        # print("Escolha o subobjetivo:", game.score, subgoal)

        # if subgoal != None:
        plan, attainable = find_plan(game, subgoal) # Ache um plano
        # print("Ache um plano")

        if attainable: # Se o plano existe
            if(subgoal == None): # Se for o subobjetivo final
                target = game._calculate_uncertainty(((game.exit.x, game.exit.y)))
            else:
                target = game._calculate_uncertainty(subgoals[subgoal])

            # target = len(plan)
            # print("Se o plano existe")
            execute_plan(game, plan) # Executa o plano
            # print("Execute o plano")
            next_state = get_state(game)
            if(subgoal == None): # Se for o subobjetivo final
                # print("Se for o subobjetivo final")
                sample = (state, (game.exit.x, game.exit.y), target + final_reward, np.zeros(input_shape, dtype=int), True) # crie a amostra final
                # print("crie a amostra final")
                game.reset_game(copy.deepcopy(init_grid), init_player) # reseta o level
                # print("reseta o level")
            else: # Se não for o subobjetivo final
                # print("Se não for o subobjetivo final")
                sample = (state, subgoals[subgoal], target, next_state, False) # crie a amostra
                # print("crie a amostra")
            dataset.append(sample)
            level_samples += 1
        else: # Se o plano não existe
            # print("Se o plano não existe")
            sample = (state, subgoal, penalization, np.zeros(input_shape, dtype=int), False) # crie a amostra com penalização
            # print("crie a amostra com penalização")
            dataset.append(sample)
            level_samples += 1

        if level_samples >= data_per_level: # Se terminou de coletar as amostras
            break
        # else:
        #     print("Finished all subgoals level, starting again.")
        #     game.reset_game(copy.deepcopy(init_grid), init_player)
        #     break

    states, targets, next_states, dones = preprocess_data(dataset)
    np.save('dataset-dynamic/states-' + str(sys.argv[1]) + '.npy', np.array(states))
    np.save('dataset-dynamic/targets-' + str(sys.argv[1]) + '.npy', np.array(targets))
    np.save('dataset-dynamic/next-states-' + str(sys.argv[1]) + '.npy', np.array(next_states))
    np.save('dataset-dynamic/dones-' + str(sys.argv[1]) + '.npy', np.array(dones))

if __name__ == "__main__":
    game = Game.BoulderDash(dynamic=True)
    print("Collecting samples...")
    collect_samples(game)
