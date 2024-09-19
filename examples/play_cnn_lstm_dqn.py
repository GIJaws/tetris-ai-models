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
    env = gym.make("SimpleTetris-v0", render_mode="human")
    state, _ = env.reset()
    state = simplify_board(state)
    input_shape = (state.shape[0], state.shape[1])
    n_actions = len(ACTION_COMBINATIONS)

    model = CNNLSTMDQN(input_shape, n_actions, SEQUENCE_LENGTH).to(device)
    model.load_state_dict(torch.load("cnn_lstm_dqn.pth"))
    model.eval()

    state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = model(state_tensor).max(1)[1].view(1, 1)

        action_combination = ACTION_COMBINATIONS[action.item()]
        next_state, reward, terminated, truncated, _ = env.step(action_combination)
        total_reward += reward
        done = terminated or truncated

        next_state = simplify_board(next_state)
        state_deque.append(next_state)
        env.render()

    print(f"Game Over. Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    play()
