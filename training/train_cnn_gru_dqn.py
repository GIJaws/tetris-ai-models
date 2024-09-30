import numpy as np
import torch
import gymnasium as gym
import gym_simpletetris
import sys
import os
from typing import cast


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from gym_simpletetris.tetris.tetris_shapes import ACTION_COMBINATIONS, simplify_board
from gym_simpletetris.tetris.tetris_env import TetrisEnv
from utils.my_logging import LoggingManager
from utils.config import load_config
from agents.CNNGRUDQNAgent import CGDAgent
from utils.reward_functions import extract_temporal_feature, extract_current_feature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config_path, model_path=None):
    config = load_config(config_path)
    logger = LoggingManager(model_name=config.MODEL_NAME)

    # TODO NEED TO MOVE REWARD FUNCTION OUT OF THE LOGGER SO I CAN EASILY SWITCH IT OUT
    env: TetrisEnv = cast(
        TetrisEnv,
        logger.setup_video_recording(  # Automate video recording
            gym.make(
                config.ENV_ID,
                render_mode=config.RENDER_MODE,
                initial_level=config.INITIAL_LEVEL,
                num_lives=config.NUM_LIVES,
            ),
            video_every_n_episodes=config.VIDEO_EVERY_N_EPISODES,
        ),
    )

    # temporal_features_names = ["x_anchor", "y_anchor", "current_piece", "held_piece"] + [
    #     f"next_piece_{i}" for i in range(4)
    # ]

    # current_features_names = ["holes", "bumpiness", "score"] + [f"col_{i}_height" for i in range(10)]

    # Initialize networks
    board, next_info = env.reset()
    board_simple = simplify_board(board)

    agent = CGDAgent(
        board_simple,
        env.action_space,
        extract_temporal_feature(next_info),
        extract_current_feature(board_simple, next_info),
        config,
        device,
        model_path=model_path,
    )

    eps_threshold: float = config.EPS_START
    try:
        for episode in range(1, config.NUM_EPISODES + 1):
            done = False
            loss: float = np.nan
            episode_q_values = []
            cur_episode_steps: int = 0
            episode_cumulative_reward: float = 0.0

            next_board, next_info = env.reset()

            next_board_simple = simplify_board(next_board)
            next_temporal_feature = extract_temporal_feature(next_info)
            next_feature = extract_current_feature(board_simple, next_info)

            current_episode_action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}

            agent.reset()

            while not done:
                info = next_info
                board_simple = next_board_simple

                temporal_feature = next_temporal_feature
                feature = next_feature

                selected_action, (policy_action, eps_threshold, step_q_values, is_random_action) = agent.select_action(
                    board_simple,
                    np.array(list(temporal_feature.values())),
                    np.array(list(feature.values())),
                    env.total_steps,
                )

                episode_q_values.append(step_q_values.cpu().numpy())

                # TODO lets track the policy actions and the selected action instead of only the selected action
                current_episode_action_count[selected_action] += 1
                next_board, step_reward, terminated, truncated, next_info = env.step(
                    ACTION_COMBINATIONS[selected_action]
                )

                next_board_simple = simplify_board(next_board)
                next_temporal_feature = extract_temporal_feature(next_info)
                next_feature = extract_current_feature(board_simple, next_info)

                done = terminated or truncated
                episode_cumulative_reward += step_reward

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
                loss, grad_norms = agent.optimize_model()

                cur_episode_steps += 1
                board_simple = next_board_simple

                logger.log_every_step(
                    total_steps=env.total_steps,
                    grad_norms=grad_norms,
                    loss=loss,
                    eps_threshold=eps_threshold,
                    info=next_info,
                )

            logger.log_to_tensorboard_every_episode(
                episode,
                episode_cumulative_reward,
                cur_episode_steps,
                info["total_lines_cleared"],
                eps_threshold,
                episode_q_values,
            )
            logger.log_action_distribution_tensorboard(current_episode_action_count, episode)
            logger.log_hardware_usage_tensorboard(episode)

            if episode % config.TARGET_UPDATE == 0:
                agent.update_target_network()

            # Save the trained model every SAVE_MODEL_INTERVAL
            if episode % 100 == 0:
                agent.save_model(logger.get_model_path(episode))
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    finally:
        # Save the final model
        agent.save_model(logger.get_model_path())
        env.close()
        logger.close_logging()


if __name__ == "__main__":
    config_path = r"tetris-ai-models\config\train_cnn_gru_dqn.yaml"

    train(config_path)
