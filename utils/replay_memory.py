from collections import deque
import random


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
