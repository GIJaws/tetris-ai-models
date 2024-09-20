import logging
import psutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Optional
import os
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
from collections import deque
from utils.helpful_utils import simplify_board
from utils.reward_functions import calculate_reward, calculate_board_metrics


class ResizeVideoOutput(gym.Wrapper):
    def __init__(self, env, width, height):
        super(ResizeVideoOutput, self).__init__(env)
        self.width = width
        self.height = height
        # Initialize stats
        self.total_reward = 0
        self.total_lines_cleared = 0
        self.time_step = 0
        # For reward calculation
        self.board_history = deque(maxlen=5)
        self.lines_cleared_history = deque(maxlen=5)
        self.time_count = -1

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time_step += 1
        self.time_count += 1
        done = terminated or truncated

        # Update board and lines cleared history
        board_state = simplify_board(obs)
        self.board_history.append(board_state.copy())
        lines_cleared = info.get("lines_cleared", 0)
        self.lines_cleared_history.append(lines_cleared)

        # Calculate custom reward
        reward, reward_components = calculate_reward(
            self.board_history, self.lines_cleared_history, done, self.time_count
        )
        self.total_reward += reward

        return obs, (reward, reward_components), terminated, truncated, info

    def reset(self, **kwargs):
        self.total_reward = 0
        self.total_lines_cleared = 0
        self.time_step = 0
        self.board_history.clear()
        self.lines_cleared_history.clear()
        self.time_count = -1
        return self.env.reset(**kwargs)

    def render(self, mode="rgb_array", **kwargs):
        frame = self.env.render(**kwargs)
        font_scale = 1
        if mode == "rgb_array":

            # Resize the frame for video recording
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # if frame.shape[2] == 3:  # If RGB image
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Add text
            cv2.putText(
                frame,
                f"Reward: {self.total_reward:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Lines Cleared: {self.total_lines_cleared}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Time Step: {self.time_step}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
            )
            # Convert back to RGB (if necessary)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


class LoggingManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.model_name}_{self.timestamp}"
        self.output_dir = f"outputs/{self.experiment_name}"
        self.log_dir = f"{self.output_dir}/logs"
        self.model_dir = f"{self.output_dir}/models"
        self.tensorboard_dir = f"{self.output_dir}/tensorboard"

        self.writer: Optional[SummaryWriter] = None
        self.logger: Optional[logging.Logger] = None
        self.logger_name: str = f"tetris_ai_training_{self.experiment_name}"
        self.setup_logging()

    def setup_logging(self):
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.tensorboard_dir)

        # Set up logging
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        log_file = f"{self.log_dir}/training.log"
        handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
        handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console_handler)

    def setup_video_recording(self, env, video_every_n_episodes=50, width=640, height=960):
        """
        Sets up video recording for the environment with resized video frames.
        Args:
            env: Gym environment instance.
            video_every_n_episodes: Record video every N episodes.
            width: Width of the resized video.
            height: Height of the resized video.
        Returns:
            env: Wrapped environment with video recording.
        """
        video_dir = f"{self.output_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)

        # Wrap the environment to resize video frames for recording
        env = ResizeVideoOutput(env, width, height)
        env = RecordVideo(
            env, video_folder=video_dir, episode_trigger=lambda episode_id: episode_id % video_every_n_episodes == 0
        )
        return env

    # Modify the logger to log video paths
    def log_video_recording(self, episode: int):
        video_dir = f"{self.output_dir}/videos"
        video_files = sorted(os.listdir(video_dir))
        if video_files:
            latest_video = video_files[-1]
            self.logger.info(f"Episode {episode}: Video recorded - {latest_video}")
        else:
            self.logger.warning(f"Episode {episode}: No video recorded.")

    def get_model_path(self, episode: Optional[int] = None) -> str:
        if episode:
            return f"{self.model_dir}/{self.model_name}_episode_{episode}.pth"
        return f"{self.model_dir}/{self.model_name}_final.pth"

    # File-Based Logging Methods
    def log_hardware_usage_file(self, episode: int):
        ram_usage = psutil.virtual_memory().percent
        self.logger.info(f"Episode {episode}: System RAM usage: {ram_usage}%")

        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024**3)
            self.logger.info(f"Episode {episode}: VRAM Usage: {vram_usage:.2f} GB")

    def log_episode_info_file(
        self,
        episode: int,
        episode_reward: float,
        steps: int,
        lines_cleared: int,
        epsilon: float,
        avg_loss: float,
    ):
        self.logger.info(
            f"Episode {episode}: Reward={episode_reward:.2f}, Steps={steps}, Lines Cleared={lines_cleared}, "
            f"Epsilon={epsilon:.4f}, Avg Loss={avg_loss:.6f}"
        )

    def log_action_distribution_file(self, action_count: Dict[int, int], episode: int):
        total_actions = sum(action_count.values())
        if total_actions == 0:
            self.logger.info(f"Episode {episode}: No actions taken in this logging period.")
            return
        action_dist = {k: v / total_actions for k, v in action_count.items()}
        self.logger.info(f"Episode {episode}: Action Distribution - {action_dist}")
        self.logger.info(f"Episode {episode}: Action Count - {action_count}")

    def log_q_values_file(self, episode: int, q_values: List[float], interval: int = 100):
        if episode % interval != 0 or not q_values:
            return
        avg_q_value = np.mean(q_values)
        min_q_value = np.min(q_values)
        max_q_value = np.max(q_values)
        std_q_value = np.std(q_values)
        self.logger.info(
            f"Episode {episode}: Avg Q-Value={avg_q_value:.4f}, Min Q-Value={min_q_value:.4f}, "
            f"Max Q-Value={max_q_value:.4f}, Std Dev Q-Value={std_q_value:.4f}"
        )

    def log_loss_file(self, loss: float, episode: int):
        self.logger.info(f"Episode {episode}: Loss={loss:.6f}")

    # TensorBoard Logging Methods (Called Every Episode)
    def log_to_tensorboard_every_episode(
        self,
        episode: int,
        episode_reward: float,
        steps: int,
        lines_cleared: int,
        epsilon: float,
        loss: Optional[float],
        q_values: List[float],
        reward_components: Dict[str, float],
    ):
        if self.writer is None:
            return

        # Log basic episode metrics
        self.writer.add_scalar("Episode/Reward", episode_reward, episode)
        self.writer.add_scalar("Episode/Steps", steps, episode)
        self.writer.add_scalar("Episode/Lines Cleared", lines_cleared, episode)
        self.writer.add_scalar("Episode/Epsilon", epsilon, episode)

        # Log loss if available
        if loss is not None:
            self.writer.add_scalar("Training/Loss", loss, episode)

        # Log Q-values statistics
        if q_values:
            avg_q = np.mean(q_values)
            min_q = np.min(q_values)
            max_q = np.max(q_values)
            std_q = np.std(q_values)
            self.writer.add_scalar("Q-Values/Avg", avg_q, episode)
            self.writer.add_scalar("Q-Values/Min", min_q, episode)
            self.writer.add_scalar("Q-Values/Max", max_q, episode)
            self.writer.add_scalar("Q-Values/StdDev", std_q, episode)

        # Log reward components
        for component_name, value in reward_components.items():
            self.writer.add_scalar(f"Reward Components/{component_name}", value, episode)

    # Add a method to log reward components to file
    def log_reward_components_file(self, reward_components: Dict[str, float], episode: int):
        component_strings = [f"{name}={value:.2f}" for name, value in reward_components.items()]
        components_log = ", ".join(component_strings)
        self.logger.info(f"Episode {episode}: Reward Components - {components_log}")

    def log_action_distribution_tensorboard(self, action_count: Dict[int, int], episode: int):
        if self.writer is None:
            return

        total_actions = sum(action_count.values())
        if total_actions == 0:
            return

        for action, count in action_count.items():
            frequency = count / total_actions
            self.writer.add_scalar(f"Actions/Action_{action}", frequency, episode)

    def log_hardware_usage_tensorboard(self, episode: int):
        if self.writer is None:
            return

        ram_usage = psutil.virtual_memory().percent
        self.writer.add_scalar("Hardware/RAM Usage", ram_usage, episode)

        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024**3)
            self.writer.add_scalar("Hardware/VRAM Usage", vram_usage, episode)

    # Unified Logging Method for Each Episode
    def log_every_episode(
        self,
        episode: int,
        episode_reward: float,
        steps: int,
        lines_cleared: int,
        epsilon: float,
        loss: Optional[float],
        q_values: List[float],
        action_count: Dict[int, int],
        reward_components: Dict[str, float],
        log_interval=10,
    ):
        # Log to files periodically
        if episode % log_interval == 0:
            self.log_episode_info_file(episode, episode_reward, steps, lines_cleared, epsilon, loss)
        if episode % log_interval == 0:
            self.log_q_values_file(episode, q_values)
        if episode % log_interval == 0:
            self.log_action_distribution_file(action_count, episode)
        if episode % log_interval == 0 and loss is not None:
            self.log_loss_file(loss, episode)
        if episode % log_interval == 0:
            self.log_hardware_usage_file(episode)
        if episode % log_interval == 0:
            self.log_reward_components_file(reward_components, episode)  # Log reward components to file

        # Log to TensorBoard every episode
        self.log_to_tensorboard_every_episode(
            episode, episode_reward, steps, lines_cleared, epsilon, loss, q_values, reward_components
        )
        self.log_action_distribution_tensorboard(action_count, episode)
        self.log_hardware_usage_tensorboard(episode)

    def close_logging(self):
        if self.writer:
            self.writer.close()
