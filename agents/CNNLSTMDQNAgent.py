from typing import cast
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque
from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.replay_memory import ReplayMemory
from agents.base_agent import TetrisAgent
from utils.reward_functions import calculate_board_inputs
from gym_simpletetris.tetris.tetris_shapes import simplify_board


def select_action(
    state, policy_net, steps_done: int, n_actions: int, eps_start: float, eps_end: float, eps_decay: float
):  # -> tuple[Any | int, Any | int, Any, Any, Any]:# -> tuple[Any | int, Any | int, Any, Any, Any]:# -> tuple[Any | int, Any | int, Any, Any, Any]:# -> tuple[Any | int, Any | int, Any, Any, Any]:# -> tuple[Any | int, Any | int, Any, Any, Any]:# -> tuple[Any | int, Any | int, Any, Any, Any]:# -> tuple[Any | int, Any | int, Any, Any, Any]:
    """Selects an action using epsilon-greedy policy.

    Args:
        state: Observation to select action from.
        policy_net: Neural network to use for selecting the action.
        steps_done: Number of steps done so far.
        n_actions: Number of actions to select from.
        eps_start: Starting value for epsilon.
        eps_end: Ending value for epsilon.
        eps_decay: Decay rate for epsilon.

    Returns:
        tuple: Tuple containing the selected action, the action selected by the policy net,
        the epsilon threshold, the q values from the policy net, and a boolean indicating whether
        the selected action is a random action.
    """
    sample = random.random()

    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    with torch.no_grad():
        q_values = policy_net(state)
        policy_action: int = cast(int, q_values.max(-1)[1].item())

    is_random_action = sample < eps_threshold

    selected_action: int = policy_action if is_random_action else random.randrange(n_actions)
    return selected_action, policy_action, eps_threshold, q_values, is_random_action


def compute_gradient_norm(model) -> float:
    total_norm: float = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


class CNNLSTMDQNAgent(TetrisAgent):
    def __init__(self, state_simple, input_shape, action_space, config, device):
        self.device = device
        self.config = config
        self.n_actions = action_space.n

        self.policy_net = CNNLSTMDQN(input_shape, self.n_actions, n_features=41).to(device)
        self.target_net = CNNLSTMDQN(input_shape, self.n_actions, n_features=41).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=float(self.config.LEARNING_RATE),
            weight_decay=float(self.config.WEIGHT_DECAY),
        )
        self.memory = ReplayMemory(self.config.MEMORY_SIZE)

        self.state_deque = deque([state_simple] * self.config.SEQUENCE_LENGTH, maxlen=self.config.SEQUENCE_LENGTH)

    def select_action(self, state, info, total_steps_done):
        state_simple = simplify_board(state)
        combined_state: tuple[torch.Tensor, torch.Tensor] = self._prepare_input(
            list(calculate_board_inputs(state_simple, info, num_actions=self.config.SEQUENCE_LENGTH).values())
        )

        selected_action, policy_action, eps_threshold, q_values, is_random_action = select_action(
            combined_state,
            self.policy_net,
            total_steps_done,
            self.n_actions,
            self.config.EPS_START,
            self.config.EPS_END,
            self.config.EPS_DECAY,
        )
        return selected_action, (policy_action, eps_threshold, q_values, is_random_action)

    def update(self, state_simple, action, reward, next_state_simple, done, info, next_info):

        combined_state: tuple[torch.Tensor, torch.Tensor] = self._prepare_input(
            list(calculate_board_inputs(state_simple, info, num_actions=self.config.SEQUENCE_LENGTH).values())
        )

        next_combined_state: tuple[torch.Tensor, torch.Tensor] = self._prepare_input(
            list(
                calculate_board_inputs(next_state_simple, next_info, num_actions=self.config.SEQUENCE_LENGTH).values()
            )
        )

        self.state_deque.append(next_state_simple)
        self.memory.push(
            combined_state,
            torch.tensor([[action]], device=self.device, dtype=torch.long),
            reward,
            next_combined_state,
            done,
        )

    def save_model(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def reset(self, state_simple):
        self.state_deque.clear()
        self.state_deque.extend([state_simple] * self.config.SEQUENCE_LENGTH)

    def optimize_model(self) -> tuple[float, tuple[float, float]]:
        if len(self.memory) < self.config.BATCH_SIZE:
            return np.nan, (np.nan, np.nan)  # No loss to report

        transitions = self.memory.sample(self.config.BATCH_SIZE)
        batch = list(zip(*transitions))

        state_features = batch[0]
        state_batch, features_batch = zip(*state_features)
        state_batch = torch.cat(state_batch)
        features_batch = torch.cat(features_batch)

        next_state_features = batch[3]
        next_state_batch, next_features_batch = zip(*next_state_features)
        next_state_batch = torch.cat(next_state_batch)
        next_features_batch = torch.cat(next_features_batch)

        action_batch = torch.cat(batch[1])
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.bool, device=self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net((state_batch, features_batch)).gather(1, action_batch)

        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
            non_final_mask = ~done_batch
            if non_final_mask.sum() > 0:
                next_state_values[non_final_mask] = self.target_net(
                    (next_state_batch[non_final_mask], next_features_batch[non_final_mask])
                ).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm before clipping
        grad_norm_before = compute_gradient_norm(self.policy_net)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)

        # Calculate gradient norm after clipping
        grad_norm_after = compute_gradient_norm(self.policy_net)
        self.optimizer.step()

        return loss.item(), (grad_norm_before, grad_norm_after)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _prepare_input(self, board_inputs):
        state_tensor = torch.tensor(np.array(self.state_deque), dtype=torch.float32, device=self.device).unsqueeze(0)

        features_tensor = torch.tensor([list(board_inputs)], dtype=torch.float32, device=self.device)
        return state_tensor, features_tensor
