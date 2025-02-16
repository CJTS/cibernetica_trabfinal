import pygame
import Game
import tensorflow as tf
import numpy as np
from Helper import get_state, input_shape
from DQNAgent import DQN
import torch
from Parser import file_to_grid
import heapq
import time
import random
import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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
        _, died, _ = game.play_step(True)
        # time.sleep(.3)
        if len(plan) == 0:
            break
        elif died:
            return False
        elif(len(plan) > 0):
            action = plan.pop()
    return True

def exec(game, dynamic, mode):
    game.dynamic = True if dynamic == "dynamic" else False
    game.init_game()
    total_plan = 0

    start_time = time.time()
    grid, player, exit = file_to_grid('levels/lvl4.txt')
    game.grid = grid
    game.player = player
    game.exit = exit
    game.play_step()

    # new_model = tf.keras.models.load_model('boulderdash-dqn.keras')
    state_dict = torch.load('dqn.pth', mmap=True, weights_only=True)
    state_uncertainty_dict = torch.load('dqn-uncertainty.pth', mmap=True, weights_only=True)

    with torch.device('meta'):
        model = DQN()
        model_uncertainty = DQN()
    model.load_state_dict(state_dict, assign=True)
    model_uncertainty.load_state_dict(state_uncertainty_dict, assign=True)
    model.eval()
    model_uncertainty.eval()

    while not (game.score >= 5 and game.player == game.exit):
        state = get_state(game)
        subgoals = game.subgoals()
        if mode in ['DQP', 'DPQU']:

            state = np.array(state).reshape(-1, *input_shape)
            heap = []

            for subgoal in subgoals:
                state[0][subgoal[0]][subgoal[1]][6] = 1
                state_in = torch.tensor(state, dtype=torch.float32, device=device)
                qvalue_plan_len = model.forward(state_in)
                qvalue_uncertainty = model_uncertainty.forward(state_in)
                state[0][subgoal[0]][subgoal[1]][6] = 0
                if(mode == 'DQP'):
                    metric = qvalue_plan_len.item()
                else:
                    metric = qvalue_plan_len.item() * qvalue_uncertainty.item()
                heapq.heappush(heap, (metric, subgoal))

            best = heapq.heappop(heap)
            subgoal = subgoals.index(best[1]) if game.score < 5 else None
        else:
            randIndex = random.randrange(0, len(subgoals))
            subgoal = randIndex if game.score < 5 else None

        plan = game.create_plan(subgoal)
        total_plan += len(plan)
        success = execute_plan(game, plan)
        if not success:
            break

    elapsed_time = time.time() - start_time
    return not success, game.score, total_plan, elapsed_time

def plot_durations(title, attr, episode_durations):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(attr)
    plt.plot(durations_t.numpy())
    if is_ipython:
        display.display(plt.gcf())

if __name__ == '__main__':
    game = Game.BoulderDash()

    data = {
        "static": {
            "RM": {
                "died": [],
                "score": [],
                "total_plan": [],
                "elapsed_time": []
            },
            "DQP": {
                "died": [],
                "score": [],
                "total_plan": [],
                "elapsed_time": []
            },
            "DPQU": {
                "died": [],
                "score": [],
                "total_plan": [],
                "elapsed_time": []
            },
        },
        "dynamic": {
            "RM": {
                "died": [],
                "score": [],
                "total_plan": [],
                "elapsed_time": []
            },
            "DQP": {
                "died": [],
                "score": [],
                "total_plan": [],
                "elapsed_time": []
            },
            "DPQU": {
                "died": [],
                "score": [],
                "total_plan": [],
                "elapsed_time": []
            },
        }
    }

    for dynamic in ["static", "dynamic"]:
        print(dynamic)
        for mode in ['RM', 'DQP', 'DPQU']:
            print(mode)
            for i in range(30):
                print(i)
                died, score, total_plan, elapsed_time = exec(game, dynamic, mode)
                data[dynamic][mode]["died"].append(died)
                data[dynamic][mode]["score"].append(score)
                data[dynamic][mode]["total_plan"].append(total_plan)
                data[dynamic][mode]["elapsed_time"].append(elapsed_time)

    for dynamic in ["static", "dynamic"]:
        for mode in ['RM', 'DQP', 'DPQU']:
            for attr in ['died', 'score', 'total_plan', 'elapsed_time']:
                plot_durations(f"Resultado {mode} {dynamic} {attr}", attr, data[dynamic][mode][attr])
                plt.ioff()
                plt.show()

    pygame.quit()
