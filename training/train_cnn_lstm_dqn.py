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
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.helpful_utils import simplify_board, ACTION_COMBINATIONS
from utils.my_logging import (
    # log_batch,
    log_episode,
    log_hardware_usage,
    aggregate_metrics,
    close_logging,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000000  # Increased for smoother decay
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-4
NUM_EPISODES = 10000
SEQUENCE_LENGTH = 4

# Logging intervals
EPISODE_LOG_INTERVAL = 5
METRICS_AGGREGATE_INTERVAL = 10
HARDWARE_LOG_INTERVAL = 10
SAVE_MODEL_INTERVAL = 100


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(memory, policy_net, target_net, optimizer, episode):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])  # Shape: [BATCH_SIZE, SEQ_LEN, H, W]
    action_batch = torch.cat(batch[1])  # Shape: [BATCH_SIZE, 1]
    reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)  # Shape: [BATCH_SIZE]
    next_state_batch = torch.cat(batch[3])  # Shape: [BATCH_SIZE, SEQ_LEN, H, W]
    done_batch = torch.tensor(batch[4], dtype=torch.bool, device=device)  # Shape: [BATCH_SIZE]

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        non_final_mask = ~done_batch
        if non_final_mask.sum() > 0:
            next_state_values[non_final_mask] = target_net(next_state_batch[non_final_mask]).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # Calculate gradient norm for logging
    total_norm = 0
    for p in policy_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5


def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(1)[1].view(1, 1).item()
            avg_q = q_values.mean().item()
        return action, eps_threshold, avg_q
    else:
        return random.randrange(n_actions), eps_threshold, 0.0


def train():
    env = gym.make("SimpleTetris-v0")
    n_actions = len(ACTION_COMBINATIONS)

    # Initialize networks
    state, _ = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])

    policy_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0

    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    lines_cleared_list = []
    try:
        for episode in range(1, NUM_EPISODES + 1):
            state, _ = env.reset()
            state = simplify_board(state)
            state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            episode_reward = 0
            episode_steps = 0
            lines_cleared = 0
            episode_q_values = []

            while True:
                state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

                action, eps_threshold, avg_q = select_action(state_tensor, policy_net, steps_done, n_actions)

                # Execute action
                action_combination = ACTION_COMBINATIONS.get(action, ["idle"])
                next_state, reward, terminated, truncated, info = env.step(action_combination)

                next_state_simple = simplify_board(next_state)
                done = terminated or truncated

                lines_cleared += info.get("lines_cleared", 0)
                reward = calculate_reward(next_state_simple, lines_cleared, done, state)

                episode_reward += reward

                # Inside the training loop
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.max(1)[1].view(1, 1).item()
                    episode_q_values.append(q_values.mean().item())

                # Update state
                state_deque.append(next_state_simple)
                next_state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(
                    0
                )

                # Store transition in memory
                memory.push(
                    state_tensor,
                    torch.tensor([[action]], device=device, dtype=torch.long),
                    reward,
                    next_state_tensor,
                    done,
                )

                # Optimize the model
                optimize_model(memory, policy_net, target_net, optimizer, episode)

                steps_done += 1
                episode_steps += 1
                episode_q_values.append(avg_q)

                if done:
                    break

            # Log hardware usage once per episode
            if episode % HARDWARE_LOG_INTERVAL == 0:
                log_hardware_usage(episode)

            # Update target network
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Aggregate episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            lines_cleared_list.append(lines_cleared)
            # Loss tracking can be implemented here if desired

            # Log episode information at specified intervals
            if episode % EPISODE_LOG_INTERVAL == 0:
                log_episode(
                    episode,
                    episode_reward,
                    episode_steps,
                    lines_cleared,
                    eps_threshold,
                    episode_q_values,
                )

            # Aggregate and log metrics every specified interval
            if episode % METRICS_AGGREGATE_INTERVAL == 0:
                aggregate_metrics(
                    episode_rewards, episode_lengths, lines_cleared_list, interval=METRICS_AGGREGATE_INTERVAL
                )
            if episode % SAVE_MODEL_INTERVAL == 0:
                # Save the trained model every SAVE_MODEL_INTERVAL
                torch.save(policy_net.state_dict(), f"outputs/cnn_lstm_dqn_episode_{episode}.pth")
    finally:
        # Save the trained model
        torch.save(policy_net.state_dict(), "outputs/cnn_lstm_dqn.pth")
        env.close()

        # Close TensorBoard writer
        close_logging()


def calculate_reward(board, lines_cleared, game_over, last_board):
    if isinstance(board, torch.Tensor):
        board = board.squeeze().cpu().numpy()
    if isinstance(last_board, torch.Tensor):
        last_board = last_board.squeeze().cpu().numpy()

    # Ensure we're working with a 2D array
    if board.ndim == 3:
        board = np.any(board != 0, axis=2).astype(bool)
    if last_board.ndim == 3:
        last_board = np.any(last_board != 0, axis=2).astype(bool)

    reward = 0

    # Game Over Penalty
    if game_over:
        reward -= 100
        return reward

    # Line Clear Reward
    line_clear_reward = [0, 40, 100, 300, 1200]  # Standard Tetris scoring
    lines_cleared = min(lines_cleared, len(line_clear_reward) - 1)
    reward += line_clear_reward[lines_cleared]

    # Height Calculation
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])

    # Stack Height Penalty
    max_height = np.max(heights)
    reward -= 0.5 * max_height

    # Holes Calculation
    holes = 0
    for x in range(board.shape[1]):
        column = board[:, x].astype(bool)
        filled = np.where(column)[0]
        if filled.size > 0:
            # Count empty cells below the first filled cell
            holes += np.sum(~column[filled[0] :])

    reward -= 0.7 * holes

    # Bumpiness Calculation
    bumpiness = np.sum(np.abs(np.diff(heights)))
    reward -= 0.2 * bumpiness

    # Idling Penalty
    if np.array_equal(board, last_board):
        reward -= 0.1  # Small penalty for idling

    return reward


if __name__ == "__main__":
    train()
