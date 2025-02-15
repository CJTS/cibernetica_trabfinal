import numpy as np
import Game

input_shape = (30, 30, 7)  # Assuming 30x30 grid with a 7 channels

def get_state(game):
    """Return level state as one hot tensor without any subgoal."""
    tensor = np.zeros(input_shape, dtype=np.float32) # Initialize one hot tensor

    # Populate the tensor
    for x in range(Game.GRID_HEIGHT):
        for y in range(Game.GRID_WIDTH):
            tile_type = game.grid[x][y]
            if tile_type != Game.Tiles.EMPTY:
                # Set the corresponding position in the tensor to 1
                tensor[x][y][tile_type.value - 1] = 1

    # Replace with actual level initialization logic
    return tensor


def preprocess_data(dataset):
    """Prepare data for CNN training."""
    states = []
    targets = []
    next_states = []
    dones = []

    for sample in dataset:
        state, subgoal, plan_length, next_state, done = sample
        state[subgoal[0]][subgoal[1]][6] = 1
        states.append(state)
        targets.append(plan_length)
        next_states.append(next_state)
        dones.append(done)

    states = np.array(states).reshape(-1, *input_shape)
    targets = np.array(targets)
    next_states = np.array(next_states).reshape(-1, *input_shape)
    dones = np.array(dones)

    return states, targets, next_states, dones