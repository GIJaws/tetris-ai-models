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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.helpful_utils import simplify_board, ACTION_COMBINATIONS

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


def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(n_actions)


def train():
    env = gym.make("SimpleTetris-v0")
    state, _ = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])
    n_actions = len(ACTION_COMBINATIONS)

    policy_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    prev_lines = 0
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = simplify_board(state)
        last_state = state.copy()
        state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
        episode_reward = 0

        while True:
            state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

            action = select_action(state_tensor, policy_net, steps_done, n_actions)

            # action_combination is now a list of valid action indices
            action_combination = ACTION_COMBINATIONS[action]

            # Pass the list directly without wrapping it in another list
            next_state, _, terminated, truncated, info = env.step(action_combination)

            next_state_simple = simplify_board(next_state)
            done = terminated or truncated

            prev_lines -= info["lines_cleared"]
            # Calculate reward using the improved function
            reward = calculate_reward(next_state_simple, prev_lines, done, last_state)

            episode_reward += reward

            state_deque.append(next_state_simple)
            next_state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(
                state_tensor,
                torch.tensor([[action]], device=device, dtype=torch.long),
                reward,
                next_state_tensor,
                done,
            )

            optimize_model(memory, policy_net, target_net, optimizer)

            if done:
                break

            last_state = state.copy()
            state = next_state_simple
            steps_done += 1

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Reward: {episode_reward}")

    torch.save(policy_net.state_dict(), "cnn_lstm_dqn.pth")
    env.close()


def calculate_reward(board, lines_cleared, game_over, last_board):
    if isinstance(board, torch.Tensor):
        board = board.squeeze().cpu().numpy()
    if isinstance(last_board, torch.Tensor):
        last_board = last_board.squeeze().cpu().numpy()

    # Ensure we're working with a 2D array
    if board.ndim == 3:
        board = np.any(board != 0, axis=2).astype(np.bool_)
    if last_board.ndim == 3:
        last_board = np.any(last_board != 0, axis=2).astype(np.bool_)

    reward = 0

    # Game Over Penalty
    if game_over:
        reward += -100
        # You can also return early if you prefer
        return -100

    # Line Clear Reward
    line_clear_reward = [0, 40, 100, 300, 1200]  # Standard Tetris scoring
    reward += line_clear_reward[lines_cleared]

    # Height Calculation
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])

    # Stack Height Penalty
    max_height = np.max(heights)
    reward += -0.5 * max_height

    # Holes Calculation
    holes = 0
    for x in range(board.shape[1]):
        column = board[:, x].astype(np.bool_)  # Ensure the column is boolean
        filled = np.where(column)[0]
        if filled.size > 0:
            # Count empty cells below the first filled cell
            holes += np.sum(~column[filled[0] :])

    reward += -0.7 * holes

    # Bumpiness Calculation
    bumpiness = np.sum(np.abs(np.diff(heights)))
    reward += -0.2 * bumpiness

    # Idling Penalty
    if np.array_equal(board, last_board):
        reward += -0.1  # Small penalty for idling

    return reward


if __name__ == "__main__":
    train()
