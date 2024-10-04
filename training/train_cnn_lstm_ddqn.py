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
from utils.my_logging import LoggingManager
from utils.config import load_config
from agents.CNNLSTMDDQNAgent import CLDDAgent
from utils.reward_functions import extract_temporal_feature, extract_current_feature
from utils.reward_functions import calculate_reward


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

    board, next_info = env.reset()
    board_simple = next_info["simple_board"]

    agent = CLDDAgent(
        board_simple,
        env.action_space,
        extract_temporal_feature(next_info),
        extract_current_feature(board_simple, next_info),
        config,
        device,
        model_path=model_path,
    )

    eps_threshold: float = config.EPS_START
    chicken_line_sum = 0
    try:
        for episode in range(1, config.NUM_EPISODES + 1):
            done = False
            loss: float = np.nan
            episode_q_values = []
            episode_double_q_values = []
            cur_episode_steps: int = 0
            episode_cumulative_reward: float = 0.0

            next_board, next_info = env.reset()

            next_board_simple = next_info["simple_board"]
            next_temporal_feature = extract_temporal_feature(next_info)
            next_feature = extract_current_feature(board_simple, next_info)

            current_episode_action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}

            agent.reset()
            board_history = [board_simple]

            if not config.OPTIMISE_EVERY_STEP:  # TODO ensure this is the correct place to optimise once every episode
                loss, grad_norms = agent.optimize_model()

                logger.log_optimise(
                    global_step=episode,
                    grad_norms=grad_norms,
                    loss=loss,
                    eps_threshold=eps_threshold,
                )
            while not done:
                info = next_info
                board_simple = next_board_simple

                temporal_feature = next_temporal_feature
                feature = next_feature

                selected_action, (
                    policy_action,
                    eps_threshold,
                    (step_q_values, step_double_q_value),
                    is_random_action,
                ) = agent.select_action(
                    board_simple,
                    np.array(list(temporal_feature.values())),
                    np.array(list(feature.values())),
                    env.total_steps if config.OPTIMISE_EVERY_STEP else episode,
                )

                episode_q_values.append(step_q_values.cpu().numpy())
                episode_double_q_values.append(step_double_q_value)

                # TODO lets track the policy actions and the selected action instead of only the selected action
                current_episode_action_count[selected_action] += 1
                next_board, step_reward, terminated, truncated, next_info = env.step(
                    ACTION_COMBINATIONS[selected_action]
                )
                done = terminated or truncated
                next_board_simple = next_info["simple_board"]
                board_history.append(next_board_simple)

                step_reward, extra_info = calculate_reward(board_history, done, next_info)
                next_info["extra_info"] = extra_info

                next_temporal_feature = extract_temporal_feature(next_info)
                next_feature = extract_current_feature(next_board_simple, next_info)

                episode_cumulative_reward += step_reward

                agent.update(
                    (
                        info["float_board_state"],
                        np.array(list(temporal_feature.values())),
                        np.array(list(feature.values())),
                    ),
                    selected_action,
                    (
                        next_info["float_board_state"],
                        np.array(list(next_temporal_feature.values())),
                        np.array(list(next_feature.values())),
                    ),
                    step_reward,
                    done,
                    done or info["lost_a_life"],
                )
                chicken_line_sum += info["lines_cleared_per_step"]
                logger.log_every_step(total_steps=env.total_steps, info=next_info, chicken_line_sum=chicken_line_sum)
                if config.OPTIMISE_EVERY_STEP:
                    loss, grad_norms = agent.optimize_model()

                    logger.log_optimise(
                        global_step=env.total_steps,
                        grad_norms=grad_norms,
                        loss=loss,
                        eps_threshold=eps_threshold,
                    )
                cur_episode_steps += 1

            logger.log_to_tensorboard_every_episode(
                episode,
                episode_cumulative_reward,
                cur_episode_steps,
                info["total_lines_cleared"],
                eps_threshold,
                episode_q_values,
                episode_double_q_values,
                chicken_line_sum,
            )
            logger.log_action_distribution_tensorboard(current_episode_action_count, episode)
            logger.log_hardware_usage_tensorboard(episode)

            if episode % config.TARGET_UPDATE == 0:
                agent.update_target_network()

            # Save the trained model every SAVE_MODEL_INTERVAL
            if episode % config.SAVE_MODEL_INTERVAL == 0:
                agent.save_model(logger.get_model_path(episode))

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    finally:
        # Save the final model
        agent.save_model(logger.get_model_path())
        env.close()
        logger.close_logging()


if __name__ == "__main__":
    config_path = r"tetris-ai-models\config\train_cnn_lstm_ddqn.yaml"
    # model_path_str = r"outputs\cnn_lstm_double_dqn_20241003_225218\models\cnn_lstm_double_dqn_final.pth"
    model_path_str = None
    train(config_path, model_path_str)
