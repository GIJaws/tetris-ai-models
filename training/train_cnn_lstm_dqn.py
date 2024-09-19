import gymnasium as gym
import gym_simpletetris
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-4
NUM_EPISODES = 10000
SEQUENCE_LENGTH = 4


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0]).to(device)
    action_batch = torch.cat(batch[1]).to(device)
    reward_batch = torch.cat(batch[2]).to(device)
    next_state_batch = torch.cat(batch[3]).to(device)
    done_batch = torch.cat(batch[4]).to(device)

    hidden = policy_net.init_hidden(BATCH_SIZE)
    state_action_values, _ = policy_net(state_batch, hidden)
    state_action_values = state_action_values.gather(1, action_batch)

    with torch.no_grad():
        next_state_values, _ = target_net(next_state_batch, hidden)
        next_state_values = next_state_values.max(1)[0].unsqueeze(1)

    expected_state_action_values = (next_state_values * GAMMA * (1 - done_batch)) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    env = gym.make("SimpleTetris-v0")
    input_shape = (SEQUENCE_LENGTH, *env.observation_space.shape)
    n_actions = env.action_space.n

    policy_net = CNNLSTMDQN(input_shape, n_actions).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    for episode in range(NUM_EPISODES):
        state = env.reset()
        state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
        episode_reward = 0

        while True:
            state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action, _ = policy_net(state_tensor)
                    action = action.max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            state_deque.append(next_state)
            next_state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state_tensor, action, reward, next_state_tensor, done)

            optimize_model(memory, policy_net, target_net, optimizer)

            if done:
                break

            steps_done += 1

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Reward: {episode_reward}")

    torch.save(policy_net.state_dict(), "cnn_lstm_dqn.pth")
    env.close()


if __name__ == "__main__":
    train()
