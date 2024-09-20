import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
import gymnasium as gym
import gym_simpletetris
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.helpful_utils import simplify_board, ACTION_COMBINATIONS
from utils.my_logging import (
    log_episode,
    log_q_values,
    log_action_distribution,
    log_loss,
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
EPS_DECAY = 200000
TARGET_UPDATE = 100
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-5
NUM_EPISODES = 10000
SEQUENCE_LENGTH = 50

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

    # Compute Q(s_t, a) - use the last time step
    state_action_values = policy_net(state_batch)[:, -1, :].gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        non_final_mask = ~done_batch
        if non_final_mask.sum() > 0:
            next_state_values[non_final_mask] = target_net(next_state_batch[non_final_mask])[:, -1, :].max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values[:, -1, :].max(-1)[1].item()
            avg_q = q_values.mean().item()
        return action, eps_threshold, avg_q
    else:
        return random.randrange(n_actions), eps_threshold, 0.0


def train():
    env = gym.make("SimpleTetris-v0", render_mode=None)
    n_actions = len(ACTION_COMBINATIONS)

    # Initialize networks
    state, _ = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])

    policy_net = CNNLSTMDQN(input_shape, n_actions).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    memory = ReplayMemory(MEMORY_SIZE)

    # Metrics tracking
    steps_done = 0
    total_loss = 0
    loss_count = 0
    episode_q_values = []

    # Enhance logging by adding action distribution and loss tracking
    action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}
    try:
        for episode in range(1, NUM_EPISODES + 1):
            state, _ = env.reset()
            state = simplify_board(state)
            sequence_length = SEQUENCE_LENGTH  # Or any other value you choose
            state_deque = deque([state] * sequence_length, maxlen=sequence_length)
            episode_reward = 0
            episode_steps = 0
            lines_cleared = 0
            episode_q_values = []
            time_count = -1

            while True:
                time_count += 1
                state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)
                action, eps_threshold, avg_q = select_action(state_tensor, policy_net, steps_done, n_actions)

                action_count[action] += 1  # Update action counts for logging

                action_combination = ACTION_COMBINATIONS.get(action, ["idle"])
                next_state, reward, terminated, truncated, info = env.step(action_combination)
                next_state_simple = simplify_board(next_state)
                done = terminated or truncated
                lines_cleared += info.get("lines_cleared", 0)

                # Calculate and log reward components
                reward = calculate_reward(next_state_simple, lines_cleared, done, time_count)

                episode_reward += reward
                # Inside the training loop
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    last_q_values = q_values[:, -1, :]  # Use the last time step
                    episode_q_values.append(last_q_values.mean().item())

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

                loss = optimize_model(memory, policy_net, target_net, optimizer, episode)

                steps_done += 1
                episode_steps += 1

                if done:
                    break

            # Log hardware usage and update target network periodically...
            if episode % HARDWARE_LOG_INTERVAL == 0:
                log_hardware_usage(episode)

            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Log episode stats and other metrics less frequently
            if episode % EPISODE_LOG_INTERVAL == 0:
                avg_loss = total_loss / loss_count if loss_count > 0 else 0
                log_episode(
                    episode,
                    episode_reward,
                    episode_steps,
                    lines_cleared,
                    eps_threshold,
                    avg_loss,
                    interval=EPISODE_LOG_INTERVAL,
                )
                total_loss = 0
                loss_count = 0

            if episode % METRICS_AGGREGATE_INTERVAL == 0:
                log_action_distribution(action_count, episode)
                log_q_values(episode, episode_q_values, interval=METRICS_AGGREGATE_INTERVAL)
                action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}
                episode_q_values = []

            if episode % SAVE_MODEL_INTERVAL == 0:
                # Save the trained model every SAVE_MODEL_INTERVAL
                torch.save(policy_net.state_dict(), f"outputs/cnn_lstm_dqn_episode_{episode}_v5.pth")
    finally:
        torch.save(policy_net.state_dict(), "outputs/cnn_lstm_dqn_v5.pth")
        env.close()
        close_logging()


def calculate_reward(board, lines_cleared, game_over, time_count):
    reward = 0

    # Strongly reward line clears
    reward += lines_cleared * 100
    if lines_cleared > 1:
        reward += (lines_cleared - 1) * 50

    # Calculate board state
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])
    max_height = np.max(heights)
    holes = sum(np.sum(~board[:, x].astype(bool)[np.argmax(board[:, x] != 0) :]) for x in range(board.shape[1]))
    bumpiness = np.sum(np.abs(np.diff(heights)))

    # Penalties
    reward -= 0.01 * max_height
    reward -= 0.01 * holes
    reward -= 0.1 * bumpiness

    # Reward for keeping the board low
    reward += (board.shape[0] - max_height) * 0.1

    # Small reward for survival and piece placement
    reward += 0.1
    reward += time_count * 0.01

    # Large penalty for game over
    if game_over:
        reward -= 50

    return reward  # No clipping to allow for larger range


if __name__ == "__main__":
    train()
