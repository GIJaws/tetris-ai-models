import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
import gymnasium as gym
import gym_simpletetris
import sys
import os

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
LEARNING_RATE = 1e-4
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
    render_mode = "rgb_array"
    env = gym.make("SimpleTetris-v0", render_mode=render_mode)

    env = logger.setup_video_recording(env, video_every_n_episodes=50)  # Automate video recording

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

            board_history = deque(maxlen=HISTORY_LENGTH + 1)
            lines_cleared_history = deque(maxlen=HISTORY_LENGTH + 1)
            state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            # state_deque = deque(maxlen=SEQUENCE_LENGTH)
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

                lines_cleared = info["lines_cleared"]
                board_history.append(next_state_simple.copy())
                lines_cleared_history.append(lines_cleared - prev_lines_cleared)

                reward = calculate_reward(board_history, lines_cleared_history, done, time_count)

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


def calculate_reward(board_history, lines_cleared_history, game_over, time_count, window_size=5):
    """
    Calculate the reward based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as a 2D numpy array).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.
        time_count (int): Number of time steps survived.
        window_size (int): Number of recent states to consider for rolling comparison.

    Returns:
        float: Calculated reward.
    """
    reward = 0

    # 1. Survival Reward
    survival_reward = min(100, time_count * 0.1)  # Capped to prevent excessive rewards
    reward += survival_reward

    # 2. Reward for Lines Cleared
    # Calculate total lines cleared in the current step
    total_lines_cleared = sum(lines_cleared_history)
    reward += total_lines_cleared * 100  # Base reward per line
    if total_lines_cleared > 1:
        reward += (total_lines_cleared - 1) * 50  # Bonus for multiple lines

    # 3. Rolling Board Comparison
    # Only proceed if we have enough history
    if len(board_history) >= window_size:
        recent_boards = list(board_history)[-window_size:]
        cumulative_metrics_diff = {"holes": 0, "max_height": 0, "bumpiness": 0}

        # Iterate through the window to accumulate differences
        for i in range(1, window_size):
            prev_board = recent_boards[i - 1]
            current_board = recent_boards[i]

            prev_metrics = calculate_board_metrics(prev_board)
            current_metrics = calculate_board_metrics(current_board)

            # Calculate differences
            cumulative_metrics_diff["holes"] += current_metrics["holes"] - prev_metrics["holes"]
            cumulative_metrics_diff["max_height"] += current_metrics["max_height"] - prev_metrics["max_height"]
            cumulative_metrics_diff["bumpiness"] += current_metrics["bumpiness"] - prev_metrics["bumpiness"]

        # 4. Apply Weighted Rewards/Penalties based on Metrics Differences
        # Define weights (these can be tuned)
        weights = {
            "holes": -10.0,  # Penalize increase in holes
            "max_height": -5.0,  # Penalize increase in max height
            "bumpiness": -1.0,  # Penalize increase in bumpiness
        }

        # Apply penalties or rewards
        for metric, diff in cumulative_metrics_diff.items():
            if diff < 0:
                # Improvement: decrease in metric
                reward += abs(diff) * abs(weights[metric]) * 0.5  # Reward half the improvement
            elif diff > 0:
                # Deterioration: increase in metric
                reward += diff * weights[metric]  # Apply penalty

    else:
        # If not enough history, perform immediate state comparison
        if len(board_history) >= 2:
            prev_board = board_history[-2]
            current_board = board_history[-1]

            prev_metrics = calculate_board_metrics(prev_board)
            current_metrics = calculate_board_metrics(current_board)

            metrics_diff = {
                "holes": current_metrics["holes"] - prev_metrics["holes"],
                "max_height": current_metrics["max_height"] - prev_metrics["max_height"],
                "bumpiness": current_metrics["bumpiness"] - prev_metrics["bumpiness"],
            }

            # Define weights
            weights = {"holes": -10.0, "max_height": -5.0, "bumpiness": -1.0}

            # Apply penalties or rewards
            for metric, diff in metrics_diff.items():
                if diff < 0:
                    # Improvement: decrease in metric
                    reward += abs(diff) * abs(weights[metric]) * 0.5  # Reward half the improvement
                elif diff > 0:
                    # Deterioration: increase in metric
                    reward += diff * weights[metric]  # Apply penalty

    # 5. Penalty for Game Over
    if game_over:
        reward -= 500  # Significant penalty for losing the game

    return reward


def calculate_board_metrics(board):
    """
    Calculate key metrics from the board state.

    Args:
        board (np.ndarray): Simplified board with shape (width, height).

    Returns:
        dict: Metrics including max height, number of holes, and bumpiness.
    """
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])
    max_height = np.max(heights)
    holes = sum(np.sum(~board[:, x].astype(bool)[np.argmax(board[:, x] != 0) :]) for x in range(board.shape[1]))
    bumpiness = np.sum(np.abs(np.diff(heights)))

    return {"max_height": max_height, "holes": holes, "bumpiness": bumpiness}


if __name__ == "__main__":
    train()
