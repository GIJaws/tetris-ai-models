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
HISTORY_LENGTH = 100


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

    return loss.item()


def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values[:, -1, :].max(-1)[1].item()
        return action, eps_threshold, q_values[:, -1, :]
    else:
        return random.randrange(n_actions), eps_threshold, None


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
    total_steps_done = 0

    try:
        for episode in range(1, NUM_EPISODES + 1):
            state, info = env.reset()
            state = simplify_board(state)

            board_history = deque([state.copy()] * (HISTORY_LENGTH + 1), maxlen=HISTORY_LENGTH + 1)
            state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            done = False
            total_reward = 0
            loss = None
            q_values = []
            lines_cleared = 0
            episode_steps = 0

            current_episode_action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}
            time_count = -1
            while not done:
                time_count += 1
                state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

                prev_lines_cleared = lines_cleared

                action, eps_threshold, step_q_values = select_action(
                    state_tensor, policy_net, total_steps_done, n_actions
                )
                current_episode_action_count[action] += 1

                # If q-values are returned (not in the random case), append them for logging
                if step_q_values is not None:
                    # Store the q-values as numpy arrays for easier handling later
                    q_values.append(step_q_values.cpu().numpy())

                action_combination = ACTION_COMBINATIONS.get(action, [7])
                next_state, reward, terminated, truncated, _ = env.step(action_combination)
                next_state_simple = simplify_board(next_state)
                episode_steps += 1

                board_history.append(next_state_simple.copy())

                lines_cleared = info["lines_cleared"]
                reward = calculate_reward(
                    board_history, next_state_simple, lines_cleared - prev_lines_cleared, done, time_count
                )

                total_reward += reward
                done = terminated or truncated

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
                loss = optimize_model(memory, policy_net, target_net, optimizer)

                total_steps_done += 1

            # Log metrics to TensorBoard and files
            logger.log_every_episode(
                episode=episode,
                episode_reward=total_reward,
                steps=episode_steps,
                lines_cleared=info.get("lines_cleared", 0),
                epsilon=eps_threshold,
                loss=loss,
                q_values=q_values,
                action_count=current_episode_action_count,
            )

            # Update target network
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

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


def calculate_reward(board_history, board, lines_cleared, game_over, time_count):
    reward = 0

    prev_board = board_history[-1]

    # Strongly reward line clears
    reward += lines_cleared * 100
    if lines_cleared > 1:
        reward += (lines_cleared - 1) * 50

    # Calculate metrics for the previous board
    prev_heights = np.array(
        [prev_board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in prev_board.T]
    )
    prev_max_height = np.max(prev_heights)
    prev_holes = sum(
        np.sum(~prev_board[:, x].astype(bool)[np.argmax(prev_board[:, x] != 0) :]) for x in range(prev_board.shape[1])
    )
    prev_bumpiness = np.sum(np.abs(np.diff(prev_heights)))

    # Calculate metrics for the current board
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])
    max_height = np.max(heights)
    holes = sum(np.sum(~board[:, x].astype(bool)[np.argmax(board[:, x] != 0) :]) for x in range(board.shape[1]))
    bumpiness = np.sum(np.abs(np.diff(heights)))

    # Calculate differences (positive if improvement)
    height_diff = prev_max_height - max_height
    holes_diff = prev_holes - holes
    bumpiness_diff = prev_bumpiness - bumpiness

    # Reward improvements
    reward += height_diff * 1.0  # Adjust weights as needed
    reward += holes_diff * 5.0
    reward += bumpiness_diff * 0.5

    # Penalize deteriorations
    if height_diff < 0:
        reward += height_diff * 1.0  # Negative value penalizes increase in height
    if holes_diff < 0:
        reward += holes_diff * 5.0  # Negative value penalizes more holes
    if bumpiness_diff < 0:
        reward += bumpiness_diff * 0.5  # Negative value penalizes increased bumpiness

    # Small reward for survival and piece placement
    reward += 0.1
    reward += time_count * 0.01

    # Large penalty for game over
    if game_over:
        reward -= 50

    return reward


if __name__ == "__main__":
    train()
