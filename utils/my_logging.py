import logging
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import RotatingFileHandler
import numpy as np

# Initialize TensorBoard writer
writer = SummaryWriter()

# Set up logging
logger = logging.getLogger("tetris_ai_training")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("training_simple_dqn.log", maxBytes=10**6, backupCount=5)
handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console_handler)


def log_reward_components(episode, reward):
    """Log the individual components of the reward for detailed analysis."""
    logger.info(f"Episode {episode}: Reward Components - Total Reward: {reward}")


def log_q_values(episode, q_values, interval=100):
    if episode % interval != 0 or not q_values:
        return
    avg_q_value = np.mean(q_values)
    min_q_value = np.min(q_values)
    max_q_value = np.max(q_values)
    std_q_value = np.std(q_values)
    logger.info(
        f"Episode {episode}: Avg Q-Value={avg_q_value:.4f}, Min Q-Value={min_q_value:.4f}, "
        f"Max Q-Value={max_q_value:.4f}, Std Dev Q-Value={std_q_value:.4f}"
    )
    writer.add_scalar("Q-Values/Avg", avg_q_value, episode)
    writer.add_scalar("Q-Values/Min", min_q_value, episode)
    writer.add_scalar("Q-Values/Max", max_q_value, episode)
    writer.add_scalar("Q-Values/StdDev", std_q_value, episode)


def log_action_distribution(action_count, episode):
    """Log the distribution of actions taken by the agent."""
    total_actions = sum(action_count.values())
    if total_actions == 0:
        logger.info(f"Episode {episode}: No actions taken in this logging period.")
        return
    action_dist = {k: v / total_actions for k, v in action_count.items()}
    logger.info(f"Episode {episode}: Action Distribution - {action_dist}")
    for action, freq in action_dist.items():
        writer.add_scalar(f"Actions/Action_{action}", freq, episode)


def log_loss(loss, episode):
    """Log the training loss."""
    logger.info(f"Episode {episode}: Loss={loss:.6f}")
    writer.add_scalar("Loss/Training", loss, episode)


def log_episode(episode, episode_reward, steps, lines_cleared, epsilon, avg_loss, interval=10):
    """Log episode summary."""
    if episode % interval != 0:
        return

    logger.info(
        f"Episode {episode}: Reward={episode_reward:.2f}, Steps={steps}, Lines Cleared={lines_cleared}, "
        f"Epsilon={epsilon:.4f}, Avg Loss={avg_loss:.6f}"
    )

    writer.add_scalar("Episode/Reward", episode_reward, episode)
    writer.add_scalar("Episode/Steps", steps, episode)
    writer.add_scalar("Episode/Lines Cleared", lines_cleared, episode)
    writer.add_scalar("Episode/Epsilon", epsilon, episode)


def aggregate_metrics(episode_rewards, episode_lengths, lines_cleared, interval=100):
    """Aggregate metrics across multiple episodes."""
    if len(episode_rewards) < interval:
        return
    avg_reward = sum(episode_rewards[-interval:]) / interval
    avg_steps = sum(episode_lengths[-interval:]) / interval
    avg_lines = sum(lines_cleared[-interval:]) / interval
    logger.info(
        f"Last {interval} Episodes: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.2f}, "
        f"Avg Lines Cleared={avg_lines:.2f}"
    )
    writer.add_scalar("Metrics/Average Reward", avg_reward, interval)
    writer.add_scalar("Metrics/Average Steps", avg_steps, interval)
    writer.add_scalar("Metrics/Average Lines Cleared", avg_lines, interval)


def log_hardware_usage(episode):
    """Log system and hardware usage like RAM and VRAM."""
    ram_usage = psutil.virtual_memory().percent
    logger.info(f"System RAM usage: {ram_usage}%")

    if torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"VRAM Usage: {vram_usage:.2f} GB")
    log_hardware_to_tensorboard(episode)


def log_to_tensorboard(episode, episode_reward, steps, lines_cleared, epsilon, loss, q_values):
    writer.add_scalar("Episode/Reward", episode_reward, episode)
    writer.add_scalar("Episode/Steps", steps, episode)
    writer.add_scalar("Episode/Lines Cleared", lines_cleared, episode)
    writer.add_scalar("Episode/Epsilon", epsilon, episode)
    if loss is not None:
        writer.add_scalar("Training/Loss", loss, episode)
    if q_values:
        avg_q = sum(q_values) / len(q_values)
        writer.add_scalar("Q-Values/Avg Q", avg_q, episode)


def log_hardware_to_tensorboard(episode):
    ram_usage = psutil.virtual_memory().percent
    writer.add_scalar("Hardware/RAM Usage", ram_usage, episode)
    if torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated() / (1024**3)
        writer.add_scalar("Hardware/VRAM Usage", vram_usage, episode)


def close_logging():
    """Close TensorBoard writer."""
    writer.close()
