import multiprocessing as mp
from multiprocessing import Manager
import numpy as np
import torch

import sys
import os
import gymnasium as gym
import gym_simpletetris

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from gym_simpletetris.tetris.tetris_env import TetrisEnv
from agents.CNNGRUDQNAgent import CGDAgent
from models.cnn_gru import CNNGRU

from utils.config import load_config


class SharedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.manager = Manager()
        self.buffer = self.manager.list()

    def push_batch(self, batch):
        if len(self.buffer) + len(batch) >= self.capacity:
            # If adding the batch would exceed capacity, remove oldest experiences
            space_needed = len(self.buffer) + len(batch) - self.capacity
            del self.buffer[:space_needed]
        self.buffer.extend(batch)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        return batch


class SharedModelBuffer:
    def __init__(self):
        self.manager = Manager()
        self.model_params = self.manager.dict()
        self.version = mp.Value("i", 0)

    def update_model(self, model_state_dict):
        # Convert tensor values to numpy arrays for sharing
        new_state_dict = {k: v.cpu().numpy() for k, v in model_state_dict.items()}

        with self.version.get_lock():
            self.model_params.clear()
            self.model_params.update(new_state_dict)
            self.version.value += 1

    def get_latest_model(self):
        with self.version.get_lock():
            current_version = self.version.value
            state_dict = {k: torch.from_numpy(v) for k, v in self.model_params.items()}
        return state_dict, current_version


# Update the game_process and training_process functions to use SharedModelBuffer


def game_process(process_id, shared_replay_buffer, shared_model_buffer, episode_complete_event, config):
    # TODO set up a logger for each agent to track metrics and log performance
    # TODO when logging stats per step use chicken sum as tensorboard only shows a sample of points when displaying large datasets
    # TODO automate chicken sum logging metrics to tensorboard
    # TODO move reward calculations to Tetris Engine
    env = TetrisEnv()
    agent = CGDAgent()
    local_buffer = []
    last_model_version = -1

    while True:
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # TODO make sure this is getting the correct sequence length
            local_buffer.append((state, action, reward, next_state, done))
            state = next_state

        # Push the entire episode's experiences to the shared buffer
        shared_replay_buffer.push_batch(local_buffer)
        local_buffer.clear()

        # Check for model updates
        new_state_dict, new_version = shared_model_buffer.get_latest_model()
        if new_version > last_model_version:
            agent.update_model(new_state_dict)
            last_model_version = new_version

        # Signal that an episode is complete
        episode_complete_event.set()


def training_process(shared_replay_buffer, shared_model_buffer, episode_complete_event, num_game_processes, config):
    # TODO this should use an agent instead of a model
    model = CNNGRU()
    optimizer = torch.optim.Adam(model.parameters())

    while True:
        if len(shared_replay_buffer.buffer) >= config.BATCH_SIZE:
            batch = shared_replay_buffer.sample(config.BATCH_SIZE)
            loss = train_model(model, optimizer, batch)

            # Update the shared model buffer with the latest parameters
            shared_model_buffer.update_model(model.state_dict())

        # Wait for all game processes to complete an episode
        for _ in range(num_game_processes):
            episode_complete_event.wait()
            episode_complete_event.clear()


def run_multi_agent_training(config_path, num_game_processes):

    config = load_config(config_path)
    shared_replay_buffer = SharedReplayBuffer(config.MEMORY_SIZE)
    shared_model_buffer = SharedModelBuffer()
    episode_complete_event = mp.Event()

    game_processes = []
    for i in range(config.NUM_GAME_PROCESSES):
        p = mp.Process(
            target=game_process, args=(i, shared_replay_buffer, shared_model_buffer, episode_complete_event, config)
        )
        p.start()
        game_processes.append(p)

    train_proc = mp.Process(
        target=training_process,
        args=(shared_replay_buffer, shared_model_buffer, episode_complete_event, num_game_processes, config),
    )
    train_proc.start()

    for p in game_processes:
        p.join()
    train_proc.join()


if __name__ == "__main__":
    NUM_GAME_PROCESSES = 4  # Adjust based on your CPU cores and desired parallelism
    config_p = r"tetris-ai-models\config\train_multi_agent_cnn_gru.yaml"
    run_multi_agent_training(config_p, NUM_GAME_PROCESSES)
