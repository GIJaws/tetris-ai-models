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
from utils.helpful_utils import simplify_board, ACTION_COMBINATIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 4


def play():
    try:
        env = gym.make("SimpleTetris-v0", render_mode="human")
    except gym.error.Error as e:
        print(f"Error initializing environment: {e}")
        return

    state, info = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])
    n_actions = len(ACTION_COMBINATIONS)

    model = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    model_path = r"outputs\2nd run\cnn_lstm_dqn_v2.pth"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    done = False
    total_reward = 0

    try:
        while not done:
            state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.max(1)[1].view(1, 1).item()

            action_combination = ACTION_COMBINATIONS.get(action, ["idle"])
            next_state, reward, terminated, truncated, _ = env.step(action_combination)
            total_reward += reward
            done = terminated or truncated

            next_state = simplify_board(next_state)
            state_deque.append(next_state)
            env.render()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred during gameplay: {e}")
    finally:
        print(f"Game Over. Total Reward: {total_reward}")
        env.close()


if __name__ == "__main__":
    play()
