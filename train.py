import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import BoulderDash

data_per_level = 200  # Unique samples per level
total_samples = 100000  # Total training samples
levels = 500  # Training levels
input_shape = (30, 30, 7)  # Assuming 30x30 grid with a single channel

# @profile()
def create_model():
    """Create a CNN model for training."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Predicting plan length or reward
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    print("Creating model...")
    model = create_model()

    states = np.zeros((data_per_level,BoulderDash.GRID_SIZE, BoulderDash.GRID_SIZE, 7), dtype=np.float32)
    targets = np.zeros((data_per_level), dtype=np.float32)

    print("Training model...")
    for level_id in range(levels):
        print(level_id + 1)
        states = np.concatenate((states, np.load('dataset/states-' + str(level_id + 1) + '.npy')), axis=0)
        targets = np.concatenate((targets, np.load('dataset/targets-' + str(level_id + 1) + '.npy')), axis=0)

    print(len(states))
    print(len(targets))
    model.fit(states, targets, epochs=10, batch_size=32)

    print("Saving model...")
    model.save('boulderdash_cnn_model.keras')

    print("Training complete.")