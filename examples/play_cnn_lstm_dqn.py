import gymnasium as gym
import gym_simpletetris
import torch
import numpy as np
from collections import deque
import sys
import os
from typing_extensions import deprecated

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.config import load_config
from agents.CNNLSTMDQNAgent import CNNLSTMDQNAgent

SEQUENCE_LENGTH = 20
from gym_simpletetris.tetris.tetris_shapes import simplify_board, ACTION_COMBINATIONS
from utils.reward_functions import calculate_board_inputs

device = torch.device("cpu")


@deprecated("Use cnn_lstm.py instead")
def play():
    config_path = r"tetris-ai-models\config\train_cnn_lstm_dqn.yaml"
    model_path = r"outputs\cnn_lstm_dqn_20240929_012903\models\cnn_lstm_dqn_episode_300.pth"
    config = load_config(config_path)
    env = gym.make("SimpleTetris-v0", render_mode="human", initial_level=1000, num_lives=1000000000)

    state, info = env.reset()
    state_simple = simplify_board(state)
    input_shape = (state_simple.shape[0], state_simple.shape[1])
    n_actions = len(ACTION_COMBINATIONS)

    model = CNNLSTMDQN(input_shape, n_actions, 41).to(device)

    agent = CNNLSTMDQNAgent(state_simple, input_shape, env.action_space, config, device, model_path=model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    done = False

    try:
        while not done:
            selected_action, (policy_action, eps_threshold, step_q_values, is_random_action) = agent.select_action(
                state_simple, info, env.total_steps
            )

            # Calculate additional features
            board_features = calculate_board_inputs(state, info)
            features_tensor = torch.tensor(
                [list(board_features.values())],
                dtype=torch.float32,
                device=device,
            )

            combined_state = (state_tensor, features_tensor)

            with torch.no_grad():
                q_values = model(combined_state)
                action = q_values.max(-1)[1].item()

            next_state, reward, terminated, truncated, _ = env.step(ACTION_COMBINATIONS[action])
            done = terminated or truncated

            next_state = simplify_board(next_state)
            state_deque.append(next_state)
            env.render()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    # except Exception as e:
    #     print(f"An error occurred during gameplay: {e}")
    finally:
        print(f"Game Over. Total Reward: {total_reward}")
        env.close()


if __name__ == "__main__":
    play()
