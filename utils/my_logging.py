import logging
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(filename="training.log", level=logging.INFO)


# Monitor system memory (RAM)
def log_system_memory():
    memory_info = psutil.virtual_memory()
    logging.info(f"System RAM: {memory_info.percent}% used")


# Monitor VRAM (if using CUDA)
def log_vram_usage():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        logging.info(f"VRAM Usage: {vram_used:.2f} GB")


# Logging after each episode
def log_episode(episode, episode_reward, steps, lines_cleared, epsilon, q_values):
    logging.info(
        f"Episode {episode}: Reward={episode_reward}, Steps={steps}, Lines Cleared={lines_cleared}, Epsilon={epsilon}"
    )
    avg_q_value = sum(q_values) / len(q_values)
    logging.info(f"Average Q-Value: {avg_q_value:.4f}")


# Logging after every batch
def log_batch(loss, grad_norm):
    logging.info(f"Batch Loss: {loss:.6f}, Gradient Norm: {grad_norm:.6f}")


# Function to aggregate metrics over intervals
def aggregate_metrics(episode_rewards, episode_lengths, lines_cleared, interval=100):
    avg_reward = sum(episode_rewards[-interval:]) / interval
    avg_steps = sum(episode_lengths[-interval:]) / interval
    avg_lines = sum(lines_cleared[-interval:]) / interval
    logging.info(
        f"Last {interval} Episodes: Avg Reward={avg_reward}, Avg Steps={avg_steps}, Avg Lines Cleared={avg_lines}"
    )


def log_hardware_usage():
    # Log system memory usage (RAM)
    ram_usage = psutil.virtual_memory().percent
    logging.info(f"System RAM usage: {ram_usage}%")

    # Log VRAM usage (if CUDA is available)
    if torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated() / (1024**3)
        logging.info(f"VRAM Usage: {vram_usage:.2f} GB")


# Initialize TensorBoard writer
writer = SummaryWriter()


# Log episode metrics to TensorBoard
def log_to_tensorboard(episode, episode_reward, steps, lines_cleared, epsilon, loss, q_values):
    writer.add_scalar("Episode/Reward", episode_reward, episode)
    writer.add_scalar("Episode/Steps", steps, episode)
    writer.add_scalar("Episode/Lines Cleared", lines_cleared, episode)
    writer.add_scalar("Episode/Epsilon", epsilon, episode)
    writer.add_scalar("Training/Loss", loss, episode)
    writer.add_scalar("Q-Values/Avg Q", sum(q_values) / len(q_values), episode)


# Log hardware usage to TensorBoard
def log_hardware_to_tensorboard(episode):
    ram_usage = psutil.virtual_memory().percent
    writer.add_scalar("Hardware/RAM Usage", ram_usage, episode)
    if torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated() / (1024**3)
        writer.add_scalar("Hardware/VRAM Usage", vram_usage, episode)
