import logging
import psutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
from collections import deque
from gym_simpletetris.utils.helpful_utils import format_value, iterate_nested_dict
from gym_simpletetris.core.game_actions import GameAction
from utils.reward_functions import calculate_reward


# TODO log metrics for n episodes instead of just the nth episode


class ResizeVideoOutput(gym.Wrapper):
    def __init__(self, env, width, height, font):
        super(ResizeVideoOutput, self).__init__(env)
        self.width = width
        self.height = height
        # Initialize stats
        self.total_reward = 0
        self.total_lines_cleared = 0
        # For reward calculation
        self.board_history = deque(maxlen=5)  # THIS BOARD HISTORY IS TO BE ONLY USED FOR REWARD FUNCTION
        self.actions: list[int] = []
        self.font = font

        self.extra_info = {}

    def step(self, action):
        # TODO THIS IS REALLY STUPID HOW I AM OVERRIDING THE STEP METHOD TO GET THE REWARD VALUES SHOULD CREATE NEW
        # ! WRAPPER FOR THE REWARD FUNCTION STUFF AND KEEP THIS JUST FOR VIDEO OUTPUT, CAN PUT THE REQUIRED INFO IN
        # !  THE INFO dict
        self.actions = action
        obs, og_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Update board and lines cleared history
        board_state = info["game_state"].board.grid
        self.board_history.append(board_state)

        # Calculate custom reward
        # _, self.extra_info = calculate_reward(self.board_history, done, info)
        self.total_reward += og_reward
        info["extra_info"] = self.extra_info

        return obs, og_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.total_reward = 0
        self.total_lines_cleared = 0
        self.board_history.clear()
        self.extra_info = {}
        return self.env.reset(**kwargs)

    def render(self, mode="rgb_array", **kwargs):
        frame = self.env.render(**kwargs)
        font_scale = 1.2
        if mode == "rgb_array":
            # Ensure frame is in the correct format (convert to numpy array if it's not)
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)

            # Convert to BGR color space if it's in RGB
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Resize the frame for video recording
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            foo = 30
            # Add text
            cv2.putText(
                frame,
                f"Chicken Sum Reward: {self.total_reward:.3f}",
                (10, foo),
                self.font,
                font_scale,
                (255, 255, 255),
                2,
            )
            foo += 30

            for k, v in ({"    Stats": []} | self.extra_info.get("current_stats", {})).items():
                cv2.putText(
                    frame,
                    f"{k}: {format_value(v)}",
                    (10, foo),
                    self.font,
                    font_scale,
                    (255, 255, 255),
                    2,
                )
                foo += 30

            for k, v in iterate_nested_dict(self.extra_info.get("rewards", {})):
                cv2.putText(
                    frame,
                    f"{k}: {format_value(v)}",
                    (10, foo),
                    self.font,
                    font_scale,
                    (255, 255, 255),
                    2,
                )
                foo += 30
            actions_str = ", ".join(act.__name__ for act in GameAction.from_index(*self.actions))
            cv2.putText(
                frame,
                f"Action: {actions_str}",
                (10, foo),
                self.font,
                font_scale,
                (255, 255, 255),
                2,
            )

        return frame


class LoggingManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.model_name}_{self.timestamp}"
        self.output_dir = f"outputs/new/{self.experiment_name}"
        self.log_dir = f"{self.output_dir}/logs"
        self.model_dir = f"{self.output_dir}/models"
        self.tensorboard_dir = f"{self.output_dir}/tensorboard"

        self.logger_name: str = f"tetris_ai_training_{self.experiment_name}"

        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer: SummaryWriter = SummaryWriter(self.tensorboard_dir)

        # Set up logging
        self.logger: logging.Logger = logging.getLogger(self.logger_name)
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

    def setup_video_recording(
        self, env, video_every_n_episodes=50, width=1280, height=1280, font=cv2.FONT_HERSHEY_PLAIN
    ):
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
        env = ResizeVideoOutput(env, width, height, font)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: episode_id % video_every_n_episodes == 0,
        )
        return env

    def log_every_step(self, total_steps: int, info, chicken_line_sum):

        extra_info = info["extra_info"]

        for k, v in iterate_nested_dict(extra_info["current_stats"][1]):
            if isinstance(v, (list, tuple, np.ndarray, str)) or v is None:
                continue
            self.writer.add_scalar(f"StepsStats/{k}", v, total_steps)

        for k, v in iterate_nested_dict(extra_info.get("rewards", {})):
            self.writer.add_scalar(f"StepRewards/{k}", v, total_steps)

        self.writer.add_scalar("ChickenLineSum", chicken_line_sum, total_steps)

    def log_optimise(
        self,
        global_step: int,
        grad_norms: tuple[float, float],
        loss: float,
        eps_threshold: float,
    ):

        # Log gradient norms
        self.writer.add_scalar("StepsStats/GradientNormBeforeClipping", grad_norms[0], global_step)
        self.writer.add_scalar("StepsStats/GradientNormAfterClipping", grad_norms[1], global_step)

        self.writer.add_scalar("StepsStats/loss", loss, global_step)
        self.writer.add_scalar("StepsStats/eps_threshold", eps_threshold, global_step)

    def get_model_path(self, episode: int | None = None) -> str:
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
        epsilon: float,
        avg_loss: float,
    ):
        self.logger.info(
            f"Episode {episode}: Reward={episode_reward:.2f}, Steps={steps}, "
            f"Epsilon={epsilon:.4f}, Avg Loss={avg_loss:.6f}"
        )

    def log_action_distribution_file(self, action_count: dict[int, int], episode: int):
        total_actions = sum(action_count.values())
        if total_actions == 0:
            self.logger.info(f"Episode {episode}: No actions taken in this logging period.")
            return
        action_dist = {k: v / total_actions for k, v in action_count.items()}
        self.logger.info(f"{episode=}, {action_count=}, {action_dist=}")

    def log_q_values_file(self, episode: int, q_values: torch.tensor, interval: int = 100):
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
        episode_steps: int,
        lines_cleared: int,
        epsilon: float,
        q_values,
        episode_double_q_values,
        chicken_line_sum,
    ):
        if self.writer is None:
            return

        # Log basic episode metrics
        self.writer.add_scalar("Episode/Reward", episode_reward, episode)
        self.writer.add_scalar("Episode/Steps", episode_steps, episode)
        self.writer.add_scalar("Episode/chicken_line_sum", chicken_line_sum, episode)
        self.writer.add_scalar("Episode/Lines Cleared", lines_cleared, episode)
        self.writer.add_scalar("Episode/Epsilon", epsilon, episode)

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

        # Log Double Q-values statistics
        if episode_double_q_values:
            avg_double_q = np.mean(episode_double_q_values)
            min_double_q = np.min(episode_double_q_values)
            max_double_q = np.max(episode_double_q_values)
            std_double_q = np.std(episode_double_q_values)
            self.writer.add_scalar("Double Q-Values/Avg", avg_double_q, episode)
            self.writer.add_scalar("Double Q-Values/Min", min_double_q, episode)
            self.writer.add_scalar("Double Q-Values/Max", max_double_q, episode)
            self.writer.add_scalar("Double Q-Values/StdDev", std_double_q, episode)

    # Add a method to log reward components to file
    def log_reward_components_file(self, reward_components: dict[str, float], episode: int):
        component_strings = [f"{name}={value:.2f}" for name, value in reward_components.items()]
        components_log = ", ".join(component_strings)
        self.logger.info(f"Episode {episode}: Reward Components - {components_log}")

    def log_action_distribution_tensorboard(self, action_count: dict[int, int], episode: int):
        if self.writer is None:
            return

        total_actions = sum(action_count.values())
        if total_actions == 0:
            return

        for action, count in action_count.items():
            frequency = count / total_actions
            # action_str = GameAction.from_index(action).action_class.__name__
            action_str = ", ".join(act.__name__ for act in GameAction.from_index(action))

            self.writer.add_scalar(f"Actions/{action}: {action_str}", frequency, episode)

    def log_hardware_usage_tensorboard(self, episode: int):
        if self.writer is None:
            return

        ram_usage = psutil.virtual_memory().percent
        self.writer.add_scalar("Hardware/RAM Usage", ram_usage, episode)

        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024**3)
            self.writer.add_scalar("Hardware/VRAM Usage", vram_usage, episode)

    def close_logging(self):
        if self.writer:
            self.writer.close()
