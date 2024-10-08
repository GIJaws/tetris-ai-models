import numpy as np
import torch
import gymnasium as gym
import gym_simpletetris
import sys
import os
from typing import cast


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from gym_simpletetris.tetris.tetris_shapes import ACTION_COMBINATIONS
from gym_simpletetris.tetris.tetris_env import TetrisEnv
from utils.config import load_config
from agents.CNNLSTMDDQNAgent import CLDDAgent

from utils.reward_functions import extract_temporal_feature, extract_current_feature


device = torch.device("cpu")


def play(config_path, model_path):
    config = load_config(config_path)

    env: TetrisEnv = cast(
        TetrisEnv,
        gym.make(
            config.ENV_ID,
            render_mode="human",
            initial_level=1,
            num_lives=10000000000000000000000000000000000,
        ),
    )

    # Initialize networks
    _, next_info = env.reset()
    next_board_simple = next_info["simple_board"]
    agent = CLDDAgent(
        next_board_simple,
        env.action_space,
        extract_temporal_feature(next_info),
        extract_current_feature(next_board_simple, next_info),
        config,
        device,
        model_path=model_path,
    )

    next_temporal_feature = extract_temporal_feature(next_info)
    next_feature = extract_current_feature(next_board_simple, next_info)

    done = False
    try:
        while not done:
            env.render()
            board_simple = next_board_simple
            temporal_feature = next_temporal_feature
            feature = next_feature

            selected_action, (policy_action, eps_threshold, step_q_values, is_random_action) = agent.select_action(
                board_simple,
                np.array(list(temporal_feature.values())),
                np.array(list(feature.values())),
                env.total_steps,
            )

            _, step_reward, terminated, truncated, next_info = env.step(ACTION_COMBINATIONS[selected_action])
            next_board_simple = next_info["simple_board"]

            next_temporal_feature = extract_temporal_feature(next_info)
            next_feature = extract_current_feature(board_simple, next_info)

            done = terminated or truncated

            agent.update(
                (board_simple, np.array(list(temporal_feature.values())), np.array(list(feature.values()))),
                selected_action,
                (
                    next_board_simple,
                    np.array(list(next_temporal_feature.values())),
                    np.array(list(next_feature.values())),
                ),
                step_reward,
                done,
            )
    except KeyboardInterrupt:
        print("Play session interrupted by user.")

    finally:
        # Save the final model
        env.close()


if __name__ == "__main__":
    config_path = r"tetris-ai-models\config\train_cnn_lstm_ddqn.yaml"
    model_path_str = r"outputs\cnn_lstm_double_dqn_20241004_223503\models\cnn_lstm_double_dqn_episode_3500.pth"
    # model_path_str = None
    play(config_path, model_path_str)
