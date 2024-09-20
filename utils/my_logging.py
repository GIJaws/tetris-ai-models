import logging
import psutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Optional
import os
from datetime import datetime


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

    def log_action_distribution_tensorboard(self, action_count: Dict[int, int], episode: int):
        if self.writer is None:
            return

        total_actions = sum(action_count.values())
        if total_actions == 0:
            return

        for action, count in action_count.items():
            frequency = count / total_actions
            self.writer.add_scalar(f"Actions/Action_{action}", frequency, episode)

    def log_q_values_tensorboard(self, q_values: List[float], episode: int):
        if self.writer is None or not q_values:
            return

        avg_q = np.mean(q_values)
        self.writer.add_scalar("Q-Values/Average", avg_q, episode)

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
    ):
        # Log to files periodically
        self.log_episode_info_file(episode, episode_reward, steps, lines_cleared, epsilon, loss)
        if episode % 10 == 0:
            self.log_q_values_file(episode, q_values)
        if episode % 10 == 0:
            self.log_action_distribution_file(action_count, episode)
        if loss is not None:
            self.log_loss_file(loss, episode)
        if episode % 10 == 0:
            self.log_hardware_usage_file(episode)

        # Log to TensorBoard every episode
        self.log_to_tensorboard_every_episode(episode, episode_reward, steps, lines_cleared, epsilon, loss, q_values)
        self.log_action_distribution_tensorboard(action_count, episode)
        self.log_hardware_usage_tensorboard(episode)

    def close_logging(self):
        if self.writer:
            self.writer.close()
