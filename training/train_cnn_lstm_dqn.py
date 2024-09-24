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
from gym_simpletetris.tetris.helpful_utils import simplify_board, ACTION_COMBINATIONS
from utils.my_logging import LoggingManager
from utils.reward_functions import calculate_board_inputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 256  # 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 100000  # 833368
TARGET_UPDATE = 500
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-5
NUM_EPISODES = 50000
SEQUENCE_LENGTH = 10
HISTORY_LENGTH = 2

# LOGGING PARAMS
LOG_EPISODE_INTERVAL = 10

# GAME SETTINGS
INITIAL_LEVEL = 30
NUM_LIVES = 0


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

    state_features = batch[0]
    state_batch, features_batch = zip(*state_features)
    state_batch = torch.cat(state_batch)
    features_batch = torch.cat(features_batch)

    next_state_features = batch[3]
    next_state_batch, next_features_batch = zip(*next_state_features)
    next_state_batch = torch.cat(next_state_batch)
    next_features_batch = torch.cat(next_features_batch)

    action_batch = torch.cat(batch[1])
    reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
    done_batch = torch.tensor(batch[4], dtype=torch.bool, device=device)

    # Compute Q(s_t, a)
    state_action_values = policy_net((state_batch, features_batch)).gather(1, action_batch)

    # Compute V(s_{t+1})
    with torch.no_grad():
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        non_final_mask = ~done_batch
        if non_final_mask.sum() > 0:
            next_state_values[non_final_mask] = target_net(
                (next_state_batch[non_final_mask], next_features_batch[non_final_mask])
            ).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()

    return loss.item()


def select_action(state, policy_net, steps_done, n_actions):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.max(-1)[1].item()
        return action, eps_threshold, q_values
    else:
        return random.randrange(n_actions), eps_threshold, None


def train():
    logger = LoggingManager(model_name="cnn_lstm_dqn")
    render_mode = "rgb_array"
    env = gym.make("SimpleTetris-v0", render_mode=render_mode, initial_level=INITIAL_LEVEL, num_lives=NUM_LIVES)

    # TODO NEED TO MOVE REWARD FUNCTION OUT OF THE LOGGER SO I CAN EASILY SWITCH IT OUT
    env = logger.setup_video_recording(env, video_every_n_episodes=100)  # Automate video recording

    n_actions = len(ACTION_COMBINATIONS)

    # Initialize networks
    state, info = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])

    policy_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH, n_features=19).to(device)
    target_net = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH, n_features=19).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    memory = ReplayMemory(MEMORY_SIZE)

    # Metrics tracking
    total_steps_done = 0
    eps_threshold = EPS_START

    try:
        for episode in range(1, NUM_EPISODES + 1):
            state, info = env.reset()
            state = simplify_board(state)

            state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            done = False
            total_reward: float = 0.0
            loss = None
            q_values = []
            episode_steps: int = 0

            current_episode_action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}
            episode_reward_components = {}
            info = {}
            while not done:
                state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

                # Calculate additional features
                board_features = calculate_board_inputs(state, info)
                features_tensor = torch.tensor(
                    [list(board_features.values())],
                    dtype=torch.float32,
                    device=device,
                )

                combined_state = (state_tensor, features_tensor)

                action, eps_threshold, step_q_values = select_action(
                    combined_state, policy_net, total_steps_done, n_actions
                )
                current_episode_action_count[action] += 1

                if step_q_values is not None:
                    q_values.append(step_q_values.cpu().numpy())

                next_state, reward, terminated, truncated, info = env.step(ACTION_COMBINATIONS[action])
                done = terminated or truncated

                detailed_info = info.get("detailed_info", {})
                next_state = simplify_board(next_state)

                episode_steps += 1
                total_reward += reward
                # Accumulate reward components
                for key, value in detailed_info.get("rewards", {}).items():
                    episode_reward_components[key] = episode_reward_components.get(key, 0) + value

                # Update state
                state_deque.append(next_state)
                next_state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(
                    0
                )

                next_board_features = calculate_board_inputs(next_state, info)
                next_features_tensor = torch.tensor(
                    [list(next_board_features.values())],
                    dtype=torch.float32,
                    device=device,
                )

                # Store transition in memory
                memory.push(
                    (state_tensor, features_tensor),
                    torch.tensor([[action]], device=device, dtype=torch.long),
                    reward,
                    (next_state_tensor, next_features_tensor),
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
                reward_components=episode_reward_components,
                log_interval=LOG_EPISODE_INTERVAL,
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


if __name__ == "__main__":
    train()
