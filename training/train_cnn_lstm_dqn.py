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
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.helpful_utils import simplify_board, ACTION_COMBINATIONS
from utils.my_logging import LoggingManager


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


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(memory, policy_net, target_net, optimizer, episode, logger):
    if len(memory) < BATCH_SIZE:
        return None  # No loss to report

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])  # Shape: [BATCH_SIZE, SEQ_LEN, H, W]
    action_batch = torch.cat(batch[1])  # Shape: [BATCH_SIZE, 1]
    reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)  # Shape: [BATCH_SIZE]
    next_state_batch = torch.cat(batch[3])  # Shape: [BATCH_SIZE, SEQ_LEN, H, W]
    done_batch = torch.tensor(batch[4], dtype=torch.bool, device=device)  # Shape: [BATCH_SIZE]

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch)[:, -1, :].gather(1, action_batch)

    # Compute V(s_{t+1})
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

    # Log loss
    logger.log_every_episode(
        episode=episode,
        episode_reward=0,  # Not applicable here
        steps=0,  # Not applicable here
        lines_cleared=0,  # Not applicable here
        epsilon=0,  # Not applicable here
        loss=loss.item(),
        q_values=[],  # Not applicable here
        action_count={},  # Not applicable here
    )

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
    logger = LoggingManager(model_name="cnn_lstm_dqn")
    env = gym.make("SimpleTetris-v0", render_mode=None)
    n_actions = len(ACTION_COMBINATIONS)

    # Initialize networks
    state, info = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])

    policy_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    memory = ReplayMemory(MEMORY_SIZE)

    # Metrics tracking
    steps_done = 0
    episode_q_values = []

    # Enhance logging by adding action distribution and loss tracking
    action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}

    try:
        for episode in range(1, NUM_EPISODES + 1):
            state, info = env.reset()
            state = simplify_board(state)
            state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            done = False
            total_reward = 0
            loss = None
            q_values = []
            current_episode_action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}

            while not done:
                state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

                action, eps_threshold, avg_q = select_action(state_tensor, policy_net, steps_done, n_actions)
                current_episode_action_count[action] += 1

                action_combination = ACTION_COMBINATIONS.get(action, ["idle"])
                next_state, reward, terminated, truncated, _ = env.step(action_combination)
                total_reward += reward
                done = terminated or truncated

                next_state = simplify_board(next_state)
                state_deque.append(next_state)

                # Store transition in memory
                memory.push(
                    state_tensor,
                    torch.tensor([[action]], device=device, dtype=torch.long),
                    reward,
                    torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0),
                    done,
                )

                # Optimize the model
                loss = optimize_model(memory, policy_net, target_net, optimizer, episode, logger)

                steps_done += 1

            # Log metrics to TensorBoard and files
            logger.log_every_episode(
                episode=episode,
                episode_reward=total_reward,
                steps=0,  # Steps per episode can be tracked if needed
                lines_cleared=info.get("lines_cleared", 0),
                epsilon=eps_threshold,
                loss=loss,
                q_values=q_values,
                action_count=current_episode_action_count,
            )

            # Update target network
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Log hardware usage every episode
            logger.log_hardware_usage_tensorboard(episode)

            # Save the trained model every SAVE_MODEL_INTERVAL
            if episode % 100 == 0:
                torch.save(policy_net.state_dict(), logger.get_model_path(episode))

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    finally:
        # Save the final model
        torch.save(policy_net.state_dict(), logger.get_model_path())
        env.close()
        logger.close_logging()


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
