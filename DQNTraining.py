from collections import deque
import tensorflow as tf
import numpy as np
from Helper import input_shape
import glob
import time
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

EPISODES = 5000

start_time = time.time()

class CustomModel(keras.Model):
  def __init__(self, main, target, latent_dim):
    super(CustomModel, self).__init__()
    self.main = main
    self.target = target
    self.latent_dim = latent_dim
    self.gamma = 0.7

  def compile(self, main_optimizer, target_optimizer, loss_fn):
    super(CustomModel, self).compile()
    self.main_optimizer = main_optimizer
    self.target_optimizer = target_optimizer
    self.loss_fn = loss_fn

  def compute_td_targets(next_q_values, rewards, discounts):
    return tf.stop_gradient(rewards + discounts * next_q_values)

  def train_step(self, data):
    input, target = data
    state = input[0][0]
    next_state = input[0][1]
    done = input[0][2]
    target = target[0]
    target = tf.cast(target, dtype=np.float32)
    done = tf.cast(done, dtype=np.float32)

    # Sample random points in the latent space
    # batch_size = tf.shape(data)[0]
    # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

    # Calculate Q(s',g')
    calculated_min_q = self.target(next_state)
    # Calculate reward + gamma * Q(s',g')
    target_val = (target + (1. - done) * self.gamma * calculated_min_q)

    # Train the main network
    with tf.GradientTape() as tape:
      main_pred = self.main(state, training=True)
      d_loss = self.loss_fn(main_pred, target_val)

    grads = tape.gradient(d_loss, self.main.trainable_weights)
    self.main_optimizer.apply_gradients(
        zip(grads, self.main.trainable_weights)
    )

    return { "d_loss": d_loss }


class DQN:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = 1e-5
    self.memory = deque(maxlen=100_000)
    main = self.build_model()
    target = self.build_model()
    latent_dim = 128
    self.model = CustomModel(main, target, latent_dim=latent_dim)
    self.model.compile(main_optimizer=Adam(self.learning_rate), target_optimizer=Adam(self.learning_rate), loss_fn=keras.losses.MSE,)
    # self.update_target_model()

  def build_model(self):
    main = Sequential()

    main.add(Conv2D(32, kernel_size=(4, 4), strides=2, activation='relu',input_shape=input_shape))
    # main.add(BatchNormalization(trainable=False, epsilon=1e-9))
    main.add(Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu'))
    # main.add(BatchNormalization(trainable=False, epsilon=1e-9))
    main.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'))
    # main.add(BatchNormalization(trainable=False, epsilon=1e-9))
    main.add(Flatten())
    main.add(Dense(128, activation='relu'))
    # main.add(BatchNormalization(trainable=False, epsilon=1e-9))
    main.add(Dense(1, activation='linear'))

    return main

  def remember(self, state, reward, next_state, done):
    self.memory.append((state, reward, next_state, done))

  # def update_target_model(self):
    # self.target_model.set_weights(self.model.get_weights())

  def replay(self, batch_size):
    if len(self.memory) < batch_size:
      return

    replay_batch = random.sample(self.memory, batch_size)

    for state, reward, next_state, done in replay_batch:
      target_val = reward

      if not done:
        target_val = (reward + self.gamma * np.amin(self.target_model.predict(next_state, verbose=0)[0]))

      target = self.model.predict(state, verbose=0)
      target[0] = target_val
      self.model.fit(state, target, epochs=1, verbose=0)

  def save(self, path):
    self.model.main.save(path)

# class DQN(tf.keras.Model):
#   """Dense neural network class."""
#   def __init__(self):
#     super(DQN, self).__init__()
#     self.layer1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=2, activation='relu',input_shape=input_shape)
#     # self.layer2 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
#     self.layer3 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')
#     # self.layer4 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
#     self.layer5 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu')
#     # self.layer6 = tf.keras.layers.BatchNormalization(trainable=False, epsilon=1e-9)
#     self.layer7 = tf.keras.layers.Flatten()
#     self.layer8 = tf.keras.layers.Dense(128, activation='relu')
#     self.layer10 = tf.keras.layers.Dense(1, activation='linear')  # Predicting plan length or reward

#   def call(self, x):
#     """Forward pass."""
#     x = self.layer1(x)
#     # x = self.layer2(x)
#     x = self.layer3(x)
#     # x = self.layer4(x)
#     x = self.layer5(x)
#     # x = self.layer6(x)
#     x = self.layer7(x)
#     x = self.layer8(x)
#     return self.layer10(x)

#   def get_config(self):
#     """Returns the configuration of the model."""
#     config = super(DQN, self).get_config()
#     config.update({
#       "layer1": self.layer1.get_config(),
#       # "layer2": self.layer2.get_config(),
#       "layer3": self.layer3.get_config(),
#       # "layer4": self.layer4.get_config(),
#       "layer5": self.layer5.get_config(),
#       # "layer6": self.layer6.get_config(),
#       "layer7": self.layer7.get_config(),
#       "layer8": self.layer8.get_config(),
#       "layer10": self.layer10.get_config(),
#     })
#     return config

#   @classmethod
#   def from_config(cls, config):
#     """Creates a model from its config."""
#     layer1 = tf.keras.layers.Conv2D.from_config(config["layer1"])
#     # layer2 = tf.keras.layers.BatchNormalization.from_config(config["layer2"])
#     layer3 = tf.keras.layers.Conv2D.from_config(config["layer3"])
#     # layer4 = tf.keras.layers.BatchNormalization.from_config(config["layer4"])
#     layer5 = tf.keras.layers.Conv2D.from_config(config["layer5"])
#     # layer6 = tf.keras.layers.BatchNormalization.from_config(config["layer6"])
#     layer7 = tf.keras.layers.Flatten.from_config(config["layer7"])
#     layer8 = tf.keras.layers.Dense.from_config(config["layer8"])
#     layer10 = tf.keras.layers.Dense.from_config(config["layer10"])
#     model = cls()
#     model.layer1 = layer1
#     # model.layer2 = layer2
#     model.layer3 = layer3
#     # model.layer4 = layer4
#     model.layer5 = layer5
#     # model.layer6 = layer6
#     model.layer7 = layer7
#     model.layer8 = layer8
#     model.layer10 = layer10
#     return model

# class ReplayBuffer(object):
#   """Experience replay buffer that samples uniformly."""
#   def __init__(self, size):
#     self.buffer = deque(maxlen=size)

#   def add(self, state, reward, next_state, done):
#     self.buffer.append((state, reward, next_state, done))

#   def __len__(self):
#     return len(self.buffer)

#   def sample(self, num_samples):
#     states, rewards, next_states, dones = [], [], [], []
#     idx = np.random.choice(len(self.buffer), num_samples)
#     for i in idx:
#       elem = self.buffer[i]
#       state, reward, next_state, done = elem
#       states.append(np.array(state, copy=False))
#       rewards.append(reward)
#       next_states.append(np.array(next_state, copy=False))
#       dones.append(done)
#     states = np.array(states)
#     rewards = np.array(rewards, dtype=np.float32)
#     next_states = np.array(next_states)
#     dones = np.array(dones)
#     return states, rewards, next_states, dones

# @tf.function
# def train_step(states, rewards, next_states, dones):
#     """Perform a training iteration on a batch of data sampled from the experience
#     replay buffer."""
#     # Calculate targets.
#     states = tf.cast(states, dtype=np.float32)  # Convert to float
#     next_states = tf.cast(next_states, dtype=np.float32)  # Convert to float
#     rewards = tf.cast(rewards, dtype=np.float32)  # Ensure rewards are also float
#     dones = tf.cast(dones, dtype=np.float32)  # Ensure dones are also float

#     next_qs = target_nn(next_states)
#     min_next_qs = tf.reduce_min(next_qs, axis=-1)
#     target = rewards + (1. - dones) * discount * min_next_qs
#     with tf.GradientTape() as tape:
#       qs = main_nn(states)
#       loss = mse(qs, target)

#     grads = tape.gradient(loss, main_nn.trainable_variables)
#     optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

#     return loss

# main_nn = DQN()
# target_nn = DQN()

# optimizer = tf.keras.optimizers.Adam(1e-5)
# mse = tf.keras.losses.MeanSquaredError()


# Hyperparameters.
# num_episodes = dataset_size
# num_iterations = 1000
# batch_size = 32
# discount = 0.7
# buffer = ReplayBuffer(100000)
# cur_frame = 0
# update_target = 10000
# epoch_size = 1000

# Progress bar settings
# bar_length = 40  # Adjust the length of the progress bar

# print("Size of dateset:", num_episodes)

# def fit():
#   for episode in range(num_episodes):
#     state = states[episode]
#     next_state = next_states[episode]
#     reward = targets[episode]
#     done = dones[episode]
#     # Save to experience replay.
#     buffer.add(state, reward, next_state, done)

#   for iteration in range(num_iterations):
#     # Copy main_nn weights to target_nn.
#     if iteration % update_target == 0:
#       target_nn.set_weights(main_nn.get_weights())

#     # Train neural network.
#     states_train, rewards_train, next_states_train, dones_train = buffer.sample(batch_size)
#     # loss = train_step(states_train, rewards_train, next_states_train, dones_train)
#     # loss_value = loss.numpy()

#     if iteration % epoch_size == 0:
#       elapsed_time = time.time() - start_time
#       avg_time_per_iteration = elapsed_time / (iteration + 1)
#       remaining_time = avg_time_per_iteration * (num_iterations - iteration)
#       sys.stdout.write("\033[F\033[K")  # Move up 1 lines and clear both
#       sys.stdout.write(
#         f"\rIteration: {iteration}/{num_iterations} "
#         # f"- Loss: {loss_value:.6f} "
#         f"- Elapsed: {elapsed_time:.2f}s "
#         f"- Remaining: {remaining_time:.2f}s\n"
#       )
#       sys.stdout.flush()

#     # Progress bar calculation
#     progress = ((iteration + 1) % epoch_size) / epoch_size
#     filled_length = int(bar_length * progress)
#     bar = "█" * filled_length + "-" * (bar_length - filled_length)
#     percentage = progress * 100
#     sys.stdout.write(
#       f"\r[{bar}] {percentage:.2f}%"
#     )
#     sys.stdout.flush()
#   elapsed_time = time.time() - start_time
#   avg_time_per_iteration = elapsed_time / (iteration + 1)
#   remaining_time = avg_time_per_iteration * (num_iterations - iteration)
#   sys.stdout.write(
#     f"\rIteration: {iteration}/{num_iterations} "
#     # f"- Loss: {loss_value:.6f} "
#     f"- Elapsed: {elapsed_time:.2f}s "
#     f"- Remaining: {remaining_time:.2f}s\n"
#   )
#   sys.stdout.flush()

# fit()
# print("\nSaving model...")
# main_nn.save('dqn.keras')

if __name__ == "__main__":
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

  state_size = (30, 30, 7)
  action_size = 1
  agent = DQN(state_size, action_size)
  batch_size = 32
  scores = []
  input = []
  output = []

  for e in range(EPISODES):
    score = 0
    state = states[e]
    state = np.reshape(state, [1, *state_size])
    next_state = next_states[e]
    next_state = np.reshape(next_state, [1, *state_size])
    reward = targets[e]
    done = dones[e]
    input.append([state, next_state, done])
    output.append(reward)

  #   agent.remember(state, reward, next_state, done)
  #   score += reward
  #   state = next_state
    # agent.replay(batch_size)

    # if e % 1000:
    #   agent.update_target_model()
    #   scores.append(score)
    #   print("episode: {}/{}, score: {}".format(e, EPISODES, score))

    # if e % 10 == 0:
    #   agent.save("./boulderdash-dqn.keras")

  # dataset = tf.data.Dataset.from_tensor_slices(dataset)
  # dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
  agent.model.fit(input, output, epochs=10)
  agent.save("./boulderdash-dqn.keras")

  # episodes = [i for i in range(len(scores))]
  # plt.figure(1)
  # plt.title("DQN Performance")
  # plt.xlabel("Episodes")
  # plt.ylabel("Scores")
  # plt.plot(episodes, scores, linewidth=3, alpha=0.7, linestyle="--", marker='o', markersize=12, color='green', markerfacecolor='k', markeredgecolor='k')
  # if is_ipython:
  #     display.display(plt.gcf())
  # plt.ioff()
  # plt.show()