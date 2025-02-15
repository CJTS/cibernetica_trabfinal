from collections import deque
import tensorflow as tf
import numpy as np
from Helper import input_shape
import glob
import time
import sys

tf.config.run_functions_eagerly(True)

start_time = time.time()

class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self):
    super(DQN, self).__init__()
    self.layer1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=2, activation='relu',input_shape=input_shape)
    # self.layer2 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer3 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')
    # self.layer4 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer5 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu')
    # self.layer6 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
    self.layer7 = tf.keras.layers.Flatten()
    self.layer8 = tf.keras.layers.Dense(128, activation='relu')
    self.layer10 = tf.keras.layers.Dense(1, activation='linear')  # Predicting plan length or reward

  def call(self, x):
    """Forward pass."""
    x = self.layer1(x)
    # x = self.layer2(x)
    x = self.layer3(x)
    # x = self.layer4(x)
    x = self.layer5(x)
    # x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    return self.layer10(x)

  def get_config(self):
    """Returns the configuration of the model."""
    config = super(DQN, self).get_config()
    config.update({
      "layer1": self.layer1.get_config(),
      # "layer2": self.layer2.get_config(),
      "layer3": self.layer3.get_config(),
      # "layer4": self.layer4.get_config(),
      "layer5": self.layer5.get_config(),
      # "layer6": self.layer6.get_config(),
      "layer7": self.layer7.get_config(),
      "layer8": self.layer8.get_config(),
      "layer10": self.layer10.get_config(),
    })
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a model from its config."""
    layer1 = tf.keras.layers.Conv2D.from_config(config["layer1"])
    # layer2 = tf.keras.layers.BatchNormalization.from_config(config["layer2"])
    layer3 = tf.keras.layers.Conv2D.from_config(config["layer3"])
    # layer4 = tf.keras.layers.BatchNormalization.from_config(config["layer4"])
    layer5 = tf.keras.layers.Conv2D.from_config(config["layer5"])
    # layer6 = tf.keras.layers.BatchNormalization.from_config(config["layer6"])
    layer7 = tf.keras.layers.Flatten.from_config(config["layer7"])
    layer8 = tf.keras.layers.Dense.from_config(config["layer8"])
    layer10 = tf.keras.layers.Dense.from_config(config["layer10"])
    model = cls()
    model.layer1 = layer1
    # model.layer2 = layer2
    model.layer3 = layer3
    # model.layer4 = layer4
    model.layer5 = layer5
    # model.layer6 = layer6
    model.layer7 = layer7
    model.layer8 = layer8
    model.layer10 = layer10
    return model

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, reward, next_state, done):
    self.buffer.append((state, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, rewards, next_states, dones = [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones)
    return states, rewards, next_states, dones

@tf.function
def train_step(states, rewards, next_states, dones):
    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer."""
    # Calculate targets.
    states = tf.cast(states, dtype=np.float32)  # Convert to float
    next_states = tf.cast(next_states, dtype=np.float32)  # Convert to float
    rewards = tf.cast(rewards, dtype=np.float32)  # Ensure rewards are also float
    dones = tf.cast(dones, dtype=np.float32)  # Ensure dones are also float

    next_qs = target_nn(next_states)
    min_next_qs = tf.reduce_min(next_qs, axis=-1)
    target = rewards + (1. - dones) * discount * min_next_qs
    with tf.GradientTape() as tape:
      qs = main_nn(states)
      loss = mse(qs, target)

    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

    return loss

main_nn = DQN()
target_nn = DQN()

optimizer = tf.keras.optimizers.Adam(1e-5)
mse = tf.keras.losses.MeanSquaredError()

states_dataset_path = "dataset/states-*.npy"
targets_dataset_path = "dataset/targets-*.npy"
next_states_dataset_path = "dataset/next-states-*.npy"
dones_dataset_path = "dataset/dones-*.npy"

states_file_list = sorted(glob.glob(states_dataset_path))  # Sort to maintain order
targets_file_list = sorted(glob.glob(targets_dataset_path))  # Sort to maintain order
next_states_file_list = sorted(glob.glob(next_states_dataset_path))  # Sort to maintain order
dones_file_list = sorted(glob.glob(dones_dataset_path))  # Sort to maintain order

states_data = [np.load(file) for file in states_file_list]  # List of (200, 30, 30, 7) arrays
targets_data = [np.load(file) for file in targets_file_list]  # List of (200, 1) arrays
next_states_data = [np.load(file) for file in next_states_file_list]  # List of (200, 30, 30, 7) arrays
dones_data = [np.load(file) for file in dones_file_list]  # List of (200, 1) arrays

states_merged_data = np.concatenate(states_data, axis=0)  # Shape (200*500, 30, 30, 7)
targets_merged_data = np.concatenate(targets_data, axis=0)  # Shape (200*500, 1)
next_states_merged_data = np.concatenate(next_states_data, axis=0)  # Shape (200*500, 30, 30, 7)
dones_merged_data = np.concatenate(dones_data, axis=0)  # Shape (200*500, 1)
dataset_size = len(states_merged_data)

states = states_merged_data.reshape(-1, 30, 30, 7)  # Mean to merge into (30, 30, 7)
targets = targets_merged_data.reshape(-1, 1)  # Mean to merge into (1)
next_states = next_states_merged_data.reshape(-1, 30, 30, 7)  # Mean to merge into (30, 30, 7)
dones = dones_merged_data.reshape(-1, 1)  # Mean to merge into (1)

# Hyperparameters.
num_episodes = dataset_size
num_iterations = 1000
batch_size = 32
discount = 0.7
buffer = ReplayBuffer(100000)
cur_frame = 0
update_target = 10000
epoch_size = 1000

# Progress bar settings
bar_length = 40  # Adjust the length of the progress bar

print("Size of dateset:", num_episodes)

def fit():
  for episode in range(num_episodes):
    state = states[episode]
    next_state = next_states[episode]
    reward = targets[episode]
    done = dones[episode]
    # Save to experience replay.
    buffer.add(state, reward, next_state, done)

  for iteration in range(num_iterations):
    # Copy main_nn weights to target_nn.
    if iteration % update_target == 0:
      target_nn.set_weights(main_nn.get_weights())

    # Train neural network.
    states_train, rewards_train, next_states_train, dones_train = buffer.sample(batch_size)
    # loss = train_step(states_train, rewards_train, next_states_train, dones_train)
    # loss_value = loss.numpy()

    if iteration % epoch_size == 0:
      elapsed_time = time.time() - start_time
      avg_time_per_iteration = elapsed_time / (iteration + 1)
      remaining_time = avg_time_per_iteration * (num_iterations - iteration)
      sys.stdout.write("\033[F\033[K")  # Move up 1 lines and clear both
      sys.stdout.write(
        f"\rIteration: {iteration}/{num_iterations} "
        # f"- Loss: {loss_value:.6f} "
        f"- Elapsed: {elapsed_time:.2f}s "
        f"- Remaining: {remaining_time:.2f}s\n"
      )
      sys.stdout.flush()

    # Progress bar calculation
    progress = ((iteration + 1) % epoch_size) / epoch_size
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    percentage = progress * 100
    sys.stdout.write(
      f"\r[{bar}] {percentage:.2f}%"
    )
    sys.stdout.flush()
  elapsed_time = time.time() - start_time
  avg_time_per_iteration = elapsed_time / (iteration + 1)
  remaining_time = avg_time_per_iteration * (num_iterations - iteration)
  sys.stdout.write(
    f"\rIteration: {iteration}/{num_iterations} "
    # f"- Loss: {loss_value:.6f} "
    f"- Elapsed: {elapsed_time:.2f}s "
    f"- Remaining: {remaining_time:.2f}s\n"
  )
  sys.stdout.flush()

fit()
print("\nSaving model...")
main_nn.save('dqn.keras')
