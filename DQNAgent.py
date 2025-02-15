import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import glob

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_shape):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(7, out_channels=32, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-9, affine=False)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-9, affine=False)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-9, affine=False)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 64, 128)  # 4 * 4 * 64 depende do input, veja abaixo.

        self.fc2 = nn.Linear(128, 1)  # Saída única (ex: valor Q ou previsão de comprimento do plano)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Muda de (batch, 30, 30, 7) para (batch, 7, 30, 30)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        return self.fc2(x)  # Saída linear

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = 1
# Get the number of state observations
n_observations = (30, 30, 7)

policy_net = DQN(n_observations).to(device)
target_net = DQN(n_observations).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100_000)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).min(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # print(state_action_values, expected_state_action_values)
    loss = criterion(expected_state_action_values.unsqueeze(1), state_action_values)
    episode_durations.append(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 500
else:
    num_episodes = 50

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

for i_episode in range(num_episodes):
    # Store the transition in memory
    state = states[i_episode]
    # action = targets[i_episode]
    next_state = next_states[i_episode]
    reward = targets[i_episode]
    done = dones[i_episode]

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # action = torch.tensor(reward, dtype=torch.int64, device=device).unsqueeze(0)
    reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(0)

    memory.push(state, next_state, reward)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the policy network)
    loss = optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)


# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()

print("Model's state_dict:")
for param_tensor in policy_net.state_dict():
    print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())

torch.save(policy_net, 'dqn.ts')