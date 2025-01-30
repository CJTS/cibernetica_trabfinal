import numpy as np
import BoulderDash

input_shape = (30, 30, 7)  # Assuming 30x30 grid with a single channel

def initialize_level(level_id):
    """Initialize a level and return its initial state."""
    game = BoulderDash.BoulderDash()
    tensor = np.zeros((BoulderDash.GRID_SIZE, BoulderDash.GRID_SIZE, 7), dtype=np.float32)
    subgoals = []

    # Populate the tensor
    for x in range(BoulderDash.GRID_SIZE):
        for y in range(BoulderDash.GRID_SIZE):
            tile_type = game.grid[x][y]
            if tile_type != BoulderDash.Tiles.EMPTY:
                # Set the corresponding position in the tensor to 1
                tensor[x][y][tile_type.value - 1] = 1
            if tile_type == BoulderDash.Tiles.GEM:
                subgoals.append((x,y))

    # Output: tensor is now a one-hot representation of the grid
    print(tensor.shape)  # (30, 30, 7)

    # Replace with actual level initialization logic
    return tensor, subgoals  # state, list of eligible subgoals

print(initialize_level(1))