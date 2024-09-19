import logging
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import RotatingFileHandler

# Initialize TensorBoard writer
writer = SummaryWriter()

# Set up logging with rotating file handler
logger = logging.getLogger("tetris_ai_training")
logger.setLevel(logging.INFO)

# Create a rotating file handler which logs even debug messages
handler = RotatingFileHandler("training.log", maxBytes=10**6, backupCount=5)
handler.setLevel(logging.INFO)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)


def log_system_memory():
    memory_info = psutil.virtual_memory()
    logger.debug(f"System RAM: {memory_info.percent}% used")


def log_vram_usage():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        logger.debug(f"VRAM Usage: {vram_used:.2f} GB")


def log_episode(episode, episode_reward, steps, lines_cleared, epsilon, q_values, interval=10):
    if episode % interval != 0:
        return
    avg_q_value = sum(q_values) / len(q_values) if q_values else 0.0
    logger.info(
        f"Episode {episode}: Reward={episode_reward}, Steps={steps}, Lines Cleared={lines_cleared}, Epsilon={epsilon:.4f}, Avg Q-Value={avg_q_value:.4f}"
    )
    log_to_tensorboard(
        episode,
        episode_reward,
        steps,
        lines_cleared,
        epsilon,
        loss=None,  # Replace with actual loss if tracked
        q_values=q_values,
    )


def log_batch(loss, grad_norm, threshold=100.0, episode=None):
    if loss > threshold:
        logger.info(f"Batch Loss: {loss:.6f}, Gradient Norm: {grad_norm:.6f}")
    else:
        logger.debug(f"Batch Loss: {loss:.6f}, Gradient Norm: {grad_norm:.6f}")


def aggregate_metrics(episode_rewards, episode_lengths, lines_cleared, interval=100):
    if len(episode_rewards) < interval:
        return
    avg_reward = sum(episode_rewards[-interval:]) / interval
    avg_steps = sum(episode_lengths[-interval:]) / interval
    avg_lines = sum(lines_cleared[-interval:]) / interval
    logger.info(
        f"Last {interval} Episodes: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.2f}, Avg Lines Cleared={avg_lines:.2f}"
    )
    # Optionally, log to TensorBoard
    writer.add_scalar("Metrics/Average Reward", avg_reward, interval)
    writer.add_scalar("Metrics/Average Steps", avg_steps, interval)
    writer.add_scalar("Metrics/Average Lines Cleared", avg_lines, interval)


def log_hardware_usage(episode):
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
    writer.close()
