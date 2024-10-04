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

        # Process transitions if buffer is full or episode is done
        if len(self.n_step_buffer) == self.n_step or done:
            # Calculate n-step discounted reward
            R = sum([self.gamma**i * t.reward for i, t in enumerate(self.n_step_buffer)])

            # Get initial state and action from the first transition in buffer
            state, action = self.n_step_buffer[0][:2]

            # Get final next_state and done flag from the last transition in buffer
            next_state = self.n_step_buffer[-1].next_state
            done = self.n_step_buffer[-1].done

            # Add the n-step transition to the main memory
            self.memory.append(Transition(state, action, next_state, R, done))

        # If episode is done, process any remaining transitions
        if done:
            while len(self.n_step_buffer) > 1:
                # Remove the oldest transition
                self.n_step_buffer.popleft()

                # Recalculate n-step discounted reward for remaining transitions
                R = sum([self.gamma**i * t.reward for i, t in enumerate(self.n_step_buffer)])

                # Get initial state and action from the new first transition
                state, action = self.n_step_buffer[0][:2]

                # Get final next_state and done flag from the last transition
                next_state = self.n_step_buffer[-1].next_state
                done = self.n_step_buffer[-1].done

                # Add this n-step transition to the main memory
                self.memory.append(Transition(state, action, next_state, R, done))

            # Clear the buffer after processing all transitions
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NStepPrioritizedReplayMemory:
    def __init__(self, capacity, n_step, gamma, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer = deque(maxlen=n_step)
        self.position = 0

    def push(self, state, action, next_state, reward, done, td_error=None):
        """
        Push a new transition to the replay memory.

        If the n-step buffer is full or the episode is over, calculate the n-step
        discounted reward and append it to the main memory. Also set the initial
        priority of this transition to the absolute value of the TD error (if
        provided) plus the epsilon value, raised to the power of alpha.

        :param state: current state
        :param action: action taken
        :param next_state: next state
        :param reward: reward received
        :param done: whether the episode is over
        :param td_error: TD error of this transition (optional)
        """
        self.n_step_buffer.append(Transition(state, action, next_state, reward, done))

        if len(self.n_step_buffer) == self.n_step or done:
            # Calculate the n-step discounted reward
            R = sum([self.gamma**i * t.reward for i, t in enumerate(self.n_step_buffer)])

            # Get the initial state and action from the first transition
            state, action = self.n_step_buffer[0][:2]

            # Get the final next state and done flag from the last transition
            next_state = self.n_step_buffer[-1].next_state
            done = self.n_step_buffer[-1].done

            # Create a new n-step transition
            n_step_transition = Transition(state, action, next_state, R, done)

            if len(self.memory) == self.capacity:
                # Replace the oldest transition if the memory is full
                self.memory[self.position] = n_step_transition
            else:
                # Append the new transition to the memory
                self.memory.append(n_step_transition)

            # Set the initial priority of this transition
            priority = 1.0 if td_error is None else (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.priorities[self.position] = priority

            # Update the position

    def sample(self, batch_size):
        prios = self.priorities[: len(self.memory)]
        probs = prios / prios.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
