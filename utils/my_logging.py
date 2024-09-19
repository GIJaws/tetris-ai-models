import logging
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

# Set up logging with levels and format
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize TensorBoard writer
writer = SummaryWriter()


# Monitor system memory (RAM)
def log_system_memory():
    memory_info = psutil.virtual_memory()
    logging.debug(f"System RAM: {memory_info.percent}% used")  # Change to debug level


# Monitor VRAM (if using CUDA)
def log_vram_usage():
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        logging.debug(f"VRAM Usage: {vram_used:.2f} GB")  # Change to debug level


# Logging after each episode (aggregated)
def log_episode(episode, episode_reward, steps, lines_cleared, epsilon, q_values, interval=10):
    # Only log every 'interval' episodes to reduce log size
    if episode % interval == 0:
        avg_q_value = sum(q_values) / len(q_values) if q_values else 0
        logging.info(
            f"Episode {episode}: Reward={episode_reward}, Steps={steps}, Lines Cleared={lines_cleared}, "
            f"Epsilon={epsilon:.4f}, Avg Q-Value: {avg_q_value:.4f}"
        )


# Logging after every batch (conditionally log high losses)
def log_batch(loss, grad_norm, threshold=1.0):
    if loss > threshold:  # Only log when loss is above a threshold
        logging.info(f"Batch Loss: {loss:.6f}, Gradient Norm: {grad_norm:.6f}")
    else:
        logging.debug(f"Batch Loss: {loss:.6f}, Gradient Norm: {grad_norm:.6f}")


# Aggregate and log metrics over intervals
def aggregate_metrics(episode_rewards, episode_lengths, lines_cleared, interval=100):
    if len(episode_rewards) < interval:
        return
    avg_reward = sum(episode_rewards[-interval:]) / interval
    avg_steps = sum(episode_lengths[-interval:]) / interval
    avg_lines = sum(lines_cleared[-interval:]) / interval
    logging.info(
        f"Last {interval} Episodes: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.2f}, Avg Lines Cleared={avg_lines:.2f}"
    )


# Log hardware usage (less frequently)
def log_hardware_usage(episode, interval=100):
    # Log system memory usage (RAM) and VRAM every 'interval' episodes
    if episode % interval == 0:
        ram_usage = psutil.virtual_memory().percent
        logging.info(f"System RAM usage: {ram_usage}%")
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024**3)
            logging.info(f"VRAM Usage: {vram_usage:.2f} GB")


# Log hardware usage to TensorBoard (less frequently)
def log_hardware_to_tensorboard(episode, interval=100):
    if episode % interval == 0:
        ram_usage = psutil.virtual_memory().percent
        writer.add_scalar("Hardware/RAM Usage", ram_usage, episode)
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024**3)
            writer.add_scalar("Hardware/VRAM Usage", vram_usage, episode)


# Log episode metrics to TensorBoard
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
