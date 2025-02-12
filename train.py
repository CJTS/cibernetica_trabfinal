import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization, InputLayer
import Game
import glob

data_per_level = 200  # Unique samples per level
total_samples = 100000  # Total training samples
levels = 500  # Training levels
input_shape = (30, 30, 7)  # Assuming 30x30 grid with a single channel

# @profile()
def create_model():
    """Create a CNN model for training."""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, kernel_size=(4, 4), strides=2, activation='relu',input_shape=input_shape),
        BatchNormalization(trainable=False, epsilon=1e-9),
        Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu'),
        BatchNormalization(trainable=False, epsilon=1e-9),
        Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'),
        BatchNormalization(trainable=False, epsilon=1e-9),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(trainable=False, epsilon=1e-9),
        Dense(1, activation='linear')  # Predicting plan length or reward
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    print("Creating model...")
    model = create_model()

    states_dataset_path = "dataset/states-*.npy"
    targets_dataset_path = "dataset/targets-*.npy"

    states_file_list = sorted(glob.glob(states_dataset_path))  # Sort to maintain order
    targets_file_list = sorted(glob.glob(targets_dataset_path))  # Sort to maintain order

    states_data = [np.load(file) for file in states_file_list]  # List of (200, 30, 30, 7) arrays
    targets_data = [np.load(file) for file in targets_file_list]  # List of (200, 1) arrays

    states_merged_data = np.concatenate(states_data, axis=0)  # Shape (200*500, 30, 30, 7)
    targets_merged_data = np.concatenate(targets_data, axis=0)  # Shape (200*500, 1)

    states = states_merged_data.reshape(-1, 30, 30, 7)  # Mean to merge into (30, 30, 7)
    targets = targets_merged_data.reshape(-1, 1)  # Mean to merge into (1)

    print("Training model...")
    model.fit(states, targets, epochs=10, batch_size=32)

    print("Saving model...")
    model.save('boulderdash_cnn_model.keras')

    print("Training complete.")