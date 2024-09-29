import gymnasium as gym
import gym_simpletetris
import torch
import numpy as np
from collections import deque
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cnn_lstm_dqn import CNNLSTMDQN

SEQUENCE_LENGTH = 20
from gym_simpletetris.tetris.tetris_shapes import simplify_board, ACTION_COMBINATIONS
from utils.reward_functions import calculate_board_inputs

device = torch.device("cpu")


def play():
    try:
        env = gym.make("SimpleTetris-v0", render_mode="human", initial_level=1000, num_lives=1000000000)

    except gym.error.Error as e:
        print(f"Error initializing environment: {e}")
        return

    state, info = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])
    n_actions = len(ACTION_COMBINATIONS)

    model = CNNLSTMDQN(input_shape, n_actions, 41).to(device)
    model_path = r"outputs\cnn_lstm_dqn_20240928_220002\models\cnn_lstm_dqn_final.pth"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    done = False
    total_reward = 0

    try:
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
