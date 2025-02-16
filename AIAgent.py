import pygame
import Game
import tensorflow as tf
import numpy as np
from Helper import get_state, input_shape
from DQNAgent import DQN
import torch
from Parser import file_to_grid
import heapq

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

if __name__ == '__main__':
    game = Game.BoulderDash(with_ui=True)
    game.init_game()

    grid, player, exit = file_to_grid('levels/lvl1.txt')
    game.grid = grid
    game.player = player
    game.exit = exit
    game.play_step()

    # new_model = tf.keras.models.load_model('boulderdash-dqn.keras')
    state_dict = torch.load('dqn.pth', mmap=True, weights_only=True)
    with torch.device('meta'):
        model = DQN()
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    state = get_state(game)
    subgoals = game.subgoals()
    state = np.array(state).reshape(-1, *input_shape)
    heap = []

    for subgoal in subgoals:
        state[0][subgoal[0]][subgoal[1]][6] = 1
        state_in = torch.tensor(state, dtype=torch.float32, device=device)
        qvalue = model.forward(state_in)
        state[0][subgoal[0]][subgoal[1]][6] = 0
        heapq.heappush(heap, (qvalue.item(), subgoal))
    best = heapq.heappop(heap)
    print("Best subgoal", best)
    print("Index", subgoals.index(best[1]))
    plan = game.create_plan(subgoals.index(best[1]))
    print(plan)

    # game loop
    while True:
        game_over, died, score = game.play_step()

        if game_over == True:
            break

    print('Final Score', score)

    pygame.quit()
