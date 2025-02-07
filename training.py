import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import BoulderDash
import ctypes
import ff
import copy
import time

# Constants
data_per_level = 5  # Unique samples per level
total_samples = 100000  # Total training samples
levels = 5  # Training levels
input_shape = (30, 30, 7)  # Assuming 30x30 grid with a single channel

# Placeholder functions for the game environment
def initialize_level(game):
    """Initialize a level and return its initial state."""
    tensor = np.zeros((BoulderDash.GRID_SIZE, BoulderDash.GRID_SIZE, 7), dtype=np.float32)
    # subgoals = []
    # gemsIndex = 0

    # Populate the tensor
    for x in range(BoulderDash.GRID_SIZE):
        for y in range(BoulderDash.GRID_SIZE):
            tile_type = game.grid[x][y]
            if tile_type != BoulderDash.Tiles.EMPTY:
                # Set the corresponding position in the tensor to 1
                tensor[x][y][tile_type.value - 1] = 1
    game._update_ui()
    # Replace with actual level initialization logic
    return tensor, game.subgoals()  # state, list of eligible subgoals

def select_random_subgoal(game):
    """Randomly select an eligible subgoal."""
    return game.choosed_subgoal()

def find_plan(game, subgoal):
    """Find a plan from the current state to the given subgoal."""
    # Replace with planning logic (return plan and whether the subgoal is attainable)
    plan = []
    with open("problem.pddl", "w") as f:
        f.write(game.get_problem(subgoal))
    plan_result = ff.plan((ctypes.c_char_p * 7)(b"ff", b"-f", b"./problem.pddl", b"-o", b"./domain.pddl", b"-i", b"0"))
    plan = []
    i = 0
    attainable = False
    while plan_result[i]:
        attainable = True
        plan.append(plan_result[i].decode('utf-8'))  # Decode the C string to Python string
        i += 1
    plan.reverse()

    return plan, attainable

def execute_plan(game, plan):
    """Execute a plan and return the resulting state."""
    action = plan.pop()
    # print("Action: ", action)
    while action:
        if action == "MOVE-UP":
            game._move(BoulderDash.Direction.UP)
        elif action == "MOVE-DOWN":
            game._move(BoulderDash.Direction.DOWN)
        elif action == "MOVE-RIGHT":
            game._move(BoulderDash.Direction.RIGHT)
        elif action == "MOVE-LEFT":
            game._move(BoulderDash.Direction.LEFT)
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
            # print("Action: ", action)
            time.sleep(1)
        else:
            break
    return initialize_level(game)  # Example resulting state

# Dataset preparation
def collect_samples(game):
    """Collect samples for training using random exploration."""
    print("Collect samples for training using random exploration.")
    dataset = []

    for level_id in range(levels):
        print("Level: ", level_id)
        level_samples = 0
        game.init_game()
        state, subgoals = initialize_level(game)
        # init_grid = list(map(list, game.grid))
        # init_grid = [copy.deepcopy(i) for i in game.grid]
        # init_grid = copy.deepcopy(game.grid)

        init_grid = [[0 for _ in range(BoulderDash.GRID_SIZE)] for _ in range(BoulderDash.GRID_SIZE)]
        for x in range(len(init_grid)):
            for y in range(len(init_grid[x])):
                init_grid[x][y] = BoulderDash.Tiles(game.grid[x][y])

        init_player = game.player

        while level_samples < data_per_level:
            if not subgoals:
                break  # Skip if no subgoals available

            while subgoals:
                subgoal = select_random_subgoal(game)
                # print("Going for subgoal: ", subgoal)

                if subgoal != None:
                    plan, attainable = find_plan(game, subgoal)
                    # print("Plan: ", plan)
                    # input("Press Enter to continue...")

                    if attainable:
                        planLength = len(plan)
                        next_state = execute_plan(game, plan)
                        sample = (state, subgoal, planLength, next_state)
                        dataset.append(sample)
                        level_samples += 1
                        print("Data in level: ", level_samples)

                        state = next_state  # Update state
                    else:
                        sample = (state, subgoal, 0, None)
                        dataset.append(sample)
                        level_samples += 1
                        print("Data in level: ", level_samples)

                    if level_samples >= data_per_level:
                        break
                else:
                    print("Finished level, starting again.")
                    game.reset_game(copy.deepcopy(init_grid), init_player)
                    break

    return dataset[:total_samples]

# CNN model definition
def create_model():
    """Create a CNN model for training."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Predicting plan length or reward
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Data preprocessing
def preprocess_data(dataset):
    """Prepare data for CNN training."""
    states = []
    subgoals = []
    targets = []

    for sample in dataset:
        state, subgoal, plan_length, _ = sample
        states.append(state)
        subgoals.append(subgoal)
        targets.append(plan_length)

    states = np.array(states).reshape(-1, *input_shape)
    targets = np.array(targets)

    return states, targets

# Training process
if __name__ == "__main__":
    game = BoulderDash.BoulderDash()
    print("Collecting samples...")
    dataset = collect_samples(game)

    # print("Preprocessing data...")
    # states, targets = preprocess_data(dataset)

    # print("Creating model...")
    # model = create_model()

    # print("Training model...")
    # model.fit(states, targets, epochs=10, batch_size=32, validation_split=0.1)

    # print("Saving model...")
    # model.save("boulderdash_cnn_model.h5")

    # print("Training complete.")
