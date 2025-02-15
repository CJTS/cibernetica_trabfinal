import pygame
import Game
import tensorflow as tf
import numpy as np
from Helper import get_state, input_shape
from DQNTraining import DQN

if __name__ == '__main__':
    game = Game.BoulderDash(with_ui=True)
    game.init_game()

    new_model = tf.keras.models.load_model('dqn.keras')

    state = get_state(game)
    subgoals = game.subgoals()
    state = np.array(state).reshape(-1, *input_shape)

    for subgoal in subgoals:
        print("For subgoal: ",subgoal)
        state[0][subgoal[0]][subgoal[1]][6] = 1
        qvalue = new_model.predict(state)
        print("QValue is", qvalue[0][0])
        state[0][subgoal[0]][subgoal[1]][6] = 0

    # game loop
    while True:
        game_over, died, score = game.play_step()

        if game_over == True:
            break

    print('Final Score', score)

    pygame.quit()
