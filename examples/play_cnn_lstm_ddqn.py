import numpy as np
import torch
import gymnasium as gym
import gym_simpletetris
import sys
import os
from typing import cast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from gym_simpletetris.env import TetrisEnv

from utils.config import load_config
from agents.CNNLSTMDDQNAgent import CLDDAgent
from utils.reward_functions import extract_temporal_feature, extract_current_feature


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(config_path, model_path):
    # Load configuration and initialize logger
    config = load_config(config_path)
    config.EPS_START = 0.0  # Ensure no exploration during play
    config.EPS_END = 0.0
    # logger = LoggingManager(model_name=config.MODEL_NAME)

    # Set up the environment with the same parameters as in the training script
    env: TetrisEnv = cast(
        TetrisEnv,
        gym.make(
            config.ENV_ID,
            width=config.WIDTH,
            height=config.HEIGHT,
            buffer_height=config.BUFFER_HEIGHT,
            # visible_height=config.VISIBLE_HEIGHT,
            visible_height=config.HEIGHT + config.BUFFER_HEIGHT,
            render_mode="human",  # Override render mode for human visualization
            obs_type=config.OBSERVATION_TYPE,
            initial_level=config.INITIAL_LEVEL,
            num_lives=10000000000000000000000000000000000,
        ),
    )

    # Reset environment and extract initial state
    _, next_info = env.reset()
    board_simple = next_info["game_state"].board.grid

    # Initialize the agent with the same structure as in the training script
    agent = CLDDAgent(
        board_simple,
        env.action_space,
        extract_temporal_feature(next_info),
        extract_current_feature(next_info),
        config,
        device,
        model_path=model_path,
    )

    done = False
    try:
        while not done:
            env.render()
            # Get current state and features
            info = next_info
            board_simple = info["game_state"].board.grid
            temporal_feature = extract_temporal_feature(info)
            feature = extract_current_feature(info)

            # Select action using the trained agent (no exploration)
            selected_action, (
                policy_action,
                eps_threshold,
                (step_q_values, step_double_q_value),
                is_random_action,
            ) = agent.select_action(
                board_simple,
                np.array(list(temporal_feature.values())),
                np.array(list(feature.values())),
                total_steps_done=env.total_steps,
            )

            # Execute the selected action in the environment
            _, step_reward, terminated, truncated, next_info = env.step([selected_action])

            done = terminated or truncated

            # Update the agent's internal state even though we aren't training
            agent.update(
                (
                    info["game_state"].place_current_piece().board.grid,
                    np.array(list(temporal_feature.values())),
                    np.array(list(feature.values())),
                ),
                selected_action,
                (
                    next_info["game_state"].place_current_piece().board.grid,
                    np.array(list(extract_temporal_feature(next_info).values())),
                    np.array(list(extract_current_feature(next_info).values())),
                ),
                step_reward,
                done,
                done,
            )

    except KeyboardInterrupt:
        print("Play session interrupted by user.")

    finally:
        env.close()


if __name__ == "__main__":
    config_path = r"tetris-ai-models\config\train_cnn_lstm_ddqn.yaml"
    model_path_str = r"outputs\new\cnn_lstm_double_dqn_20241029_133823\models\cnn_lstm_double_dqn_episode_5500.pth"
    play(config_path, model_path_str)
