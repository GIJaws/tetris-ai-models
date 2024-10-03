import random
import numpy as np
from collections import deque, namedtuple


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayPrioritisedMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        self.priorities.append(reward * (reward > 0) + 1e-5)

    def sample(self, batch_size, alpha=0.6):
        sampled_indices = random.choices(
            range(len(self.memory)), weights=[x**alpha for x in self.priorities], k=batch_size
        )
        samples = [self.memory[i] for i in sampled_indices]
        return samples

    def __len__(self):
        return len(self.memory)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class NStepReplayMemory:
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, next_state, reward, done):
        # Append the most recent transition to the n-step buffer
        self.n_step_buffer.append(Transition(state, action, next_state, reward, done))

        # If the buffer is not yet full, return without adding to the memory
        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate the discounted reward for the n-step return
        R = sum([self.gamma**i * t.reward for i, t in enumerate(self.n_step_buffer)])

        # Take the first state and action from the buffer, and the last state and done flag
        state, action = self.n_step_buffer[0][:2]
        next_state = self.n_step_buffer[-1].next_state
        done = self.n_step_buffer[-1].done

        # Add the n-step transition to the memory
        self.memory.append(Transition(state, action, next_state, R, done))

        # If the episode is done, add the rest of the transitions in the buffer to the memory
        if done:
            while len(self.n_step_buffer) > 0:

                # Calculate the discounted reward for the new n-step return
                R = sum([self.gamma**i * t.reward for i, t in enumerate(self.n_step_buffer)])

                # Take the first state and action from the buffer, and the last state and done flag
                state, action = self.n_step_buffer[0][:2]
                next_state = self.n_step_buffer[-1].next_state
                done = self.n_step_buffer[-1].done

                # Add the n-step transition to the memory
                self.memory.append(Transition(state, action, next_state, R, done))

                self.n_step_buffer.popleft()  # Remove the oldest transition from the buffer

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
