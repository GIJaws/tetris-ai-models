import numpy as np
import torch
import gymnasium as gym
import gym_simpletetris
import sys
import os
from typing import cast
from typing_extensions import deprecated


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from gym_simpletetris.tetris.tetris_shapes import ACTION_COMBINATIONS, simplify_board
from gym_simpletetris.tetris.tetris_env import TetrisEnv
from utils.my_logging import LoggingManager
from utils.config import load_config
from agents.CNNLSTMDQNAgent import CNNLSTMDQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@deprecated("Use train_cnn_lstm_ddqn.py instead")
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

    # Initialize networks
    state, info = env.reset()
    state_simple = simplify_board(state)
    input_shape = (state_simple.shape[0], state_simple.shape[1])

    agent = CNNLSTMDQNAgent(state_simple, input_shape, env.action_space, config, device, model_path=model_path)

    eps_threshold: float = config.EPS_START
    try:
        for episode in range(1, config.NUM_EPISODES + 1):
            done = False
            loss: float = np.nan
            episode_q_values = []
            cur_episode_steps: int = 0
            episode_cumulative_reward: float = 0.0

            state, info = env.reset()
            state_simple = simplify_board(state)

            current_episode_action_count = {action: 0 for action in ACTION_COMBINATIONS.keys()}

            agent.reset(state_simple)

            while not done:
                prev_info = info
                selected_action, (policy_action, eps_threshold, step_q_values, is_random_action) = agent.select_action(
                    state_simple, info, env.total_steps
                )

                episode_q_values.append(step_q_values.cpu().numpy())

                # TODO lets track the policy actions and the selected action instead of only the selected action
                current_episode_action_count[selected_action] += 1
                next_state_simple, step_reward, terminated, truncated, info = env.step(
                    ACTION_COMBINATIONS[selected_action]
                )

                next_state_simple = simplify_board(next_state_simple)

                done = terminated or truncated
                episode_cumulative_reward += step_reward

                agent.update(state_simple, selected_action, step_reward, next_state_simple, done, prev_info, info)
                loss, grad_norms = agent.optimize_model()

                cur_episode_steps += 1
                state_simple = next_state_simple

                logger.log_every_step(
                    total_steps=env.total_steps,
                    grad_norms=grad_norms,
                    loss=loss,
                    eps_threshold=eps_threshold,
                    info=info,
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
    config_path = r"tetris-ai-models\config\train_cnn_lstm_dqn.yaml"
    model_path = r"outputs\cnn_lstm_dqn_20240929_000430\models\cnn_lstm_dqn_final.pth"

    train(config_path)
