from collections import deque
import tensorflow as tf
import numpy as np
from Helper import input_shape
import glob

class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.layer2 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=2, activation='relu',input_shape=input_shape)
    self.layer3 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer4 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')
    self.layer5 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer6 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu')
    self.layer7 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer8 = tf.keras.layers.Flatten()
    self.layer9 = tf.keras.layers.Dense(128, activation='relu')
    self.layer10 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer11 = tf.keras.layers.Dense(1, activation='linear')  # Predicting plan length or reward

  def call(self, x):
    """Forward pass."""
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    x = self.layer9(x)
    x = self.layer10(x)
    return self.layer11(x)

main_nn = DQN()
target_nn = DQN()

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, reward, next_state):
    self.buffer.append((state, reward, next_state))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, rewards, next_states = [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, reward, next_state = elem
      states.append(np.array(state, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
    states = np.array(states)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    return states, rewards, next_states

@tf.function
def train_step(states, rewards, next_states):
    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""
    # Calculate targets.
    states = tf.cast(states, dtype=tf.float32)  # Convert to float
    next_states = tf.cast(next_states, dtype=tf.float32)  # Convert to float
    rewards = tf.cast(rewards, dtype=tf.float32)  # Ensure rewards are also float

    next_qs = target_nn(next_states)
    min_next_qs = tf.reduce_min(next_qs, axis=-1)
    target = rewards + discount * min_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        loss = mse(qs, target)
        grads = tape.gradient(loss, main_nn.trainable_variables)
        optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

    return loss

# Hyperparameters.
num_episodes = 200
batch_size = 32
discount = 0.7
buffer = ReplayBuffer(100000)
cur_frame = 0

# Start training. Play game once and then train with a batch.

states_dataset_path = "dataset/states-*.npy"
targets_dataset_path = "dataset/targets-*.npy"
next_states_dataset_path = "dataset/next-states-*.npy"

states_file_list = sorted(glob.glob(states_dataset_path))  # Sort to maintain order
targets_file_list = sorted(glob.glob(targets_dataset_path))  # Sort to maintain order
next_states_file_list = sorted(glob.glob(next_states_dataset_path))  # Sort to maintain order

states_data = [np.load(file) for file in states_file_list]  # List of (200, 30, 30, 7) arrays
targets_data = [np.load(file) for file in targets_file_list]  # List of (200, 1) arrays
next_states_data = [np.load(file) for file in next_states_file_list]  # List of (200, 30, 30, 7) arrays

states_merged_data = np.concatenate(states_data, axis=0)  # Shape (200*500, 30, 30, 7)
targets_merged_data = np.concatenate(targets_data, axis=0)  # Shape (200*500, 1)
next_states_merged_data = np.concatenate(next_states_data, axis=0)  # Shape (200*500, 30, 30, 7)

states = states_merged_data.reshape(-1, 30, 30, 7)  # Mean to merge into (30, 30, 7)
targets = targets_merged_data.reshape(-1, 1)  # Mean to merge into (1)
next_states = next_states_merged_data.reshape(-1, 30, 30, 7)  # Mean to merge into (30, 30, 7)

for episode in range(num_episodes):
    state = states[episode]
    next_state = next_states[episode]
    reward = targets[episode]
    # Save to experience replay.
    buffer.add(state, reward, next_state)
    state = next_state
    cur_frame += 1

    # Copy main_nn weights to target_nn.
    if cur_frame % 2000 == 0:
        target_nn.set_weights(main_nn.get_weights())

    # Train neural network.
    if len(buffer) >= batch_size:
        states_train, rewards_train, next_states_train = buffer.sample(batch_size)
        loss = train_step(states_train, rewards_train, next_states_train)

print("Saving model...")
main_nn.save('dqn.keras')
