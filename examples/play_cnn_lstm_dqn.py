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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 4


def play():
    env = gym.make("SimpleTetris-v0", render_mode="human")
    input_shape = (SEQUENCE_LENGTH, *env.observation_space.shape)
    n_actions = env.action_space.n

    model = CNNLSTMDQN(input_shape, n_actions).to(device)
    model.load_state_dict(torch.load("cnn_lstm_dqn.pth"))
    model.eval()

    state = env.reset()
    state_deque = deque([state] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(np.array(state_deque), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _ = model(state_tensor)
            action = action.max(1)[1].view(1, 1)

        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward

        state_deque.append(next_state)
        env.render()

    print(f"Game Over. Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    play()
