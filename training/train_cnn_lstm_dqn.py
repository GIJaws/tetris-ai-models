import gymnasium as gym
import gym_simpletetris
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import math
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.helpful_utils import simplify_board

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

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
    next_state_batch = torch.cat(batch[3])
    done_batch = torch.tensor(batch[4], dtype=torch.bool, device=device)

    # Model will convert bool to float32 internally
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[~done_batch] = target_net(next_state_batch[~done_batch]).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train():
    env = gym.make("SimpleTetris-v0")
    state, _ = env.reset()
    state = simplify_board(state)
    # state shape is now (10, 40)
    input_shape = (state.shape[0], state.shape[1])
    n_actions = env.action_space.n

    policy_net = CNNLSTMDQN(input_shape, n_actions).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = simplify_board(state)
        state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
        episode_reward = 0

        while True:
            # Convert to torch tensor (float32)
            state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)
            # state_tensor shape: (1, sequence_length, 10, 40)

            action = select_action(state_tensor, policy_net, steps_done, n_actions, device=device)

            next_state, reward, terminated, truncated, _ = env.step([action.item()])
            next_state = simplify_board(next_state)
            done = terminated or truncated
            episode_reward += reward

            state_deque.append(next_state)
            next_state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

            # Store float tensors in memory
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


def select_action(state, policy_net, steps_done, n_actions, device):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


if __name__ == "__main__":
    train()
