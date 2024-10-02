from abc import ABC, abstractmethod
from typing import Any
import random
import math
import torch
from typing import cast


class TetrisAgent(ABC):
    @abstractmethod
    def select_action(self, state, temporal_feature, current_feature, total_steps_done) -> tuple[int, tuple[Any, ...]]:
        pass

    @abstractmethod
    def update(self, state, action, next_state, reward, done):
        pass

    @abstractmethod
    def optimize_model(self) -> tuple[float, tuple[float, float]]:
        pass

    @abstractmethod
    def update_target_network(self):
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def reset(self, state_simple):
        pass

    @staticmethod
    def select_action_static(
        state,
        policy_net,
        target_net,
        steps_done: int,
        n_actions: int,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
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
            policy_q_values = policy_net(state)
            target_q_values = target_net(state)

            # Double DQN: Use policy network to select action, target network to evaluate it
            policy_action: int = cast(int, policy_q_values.max(-1)[1].item())
            double_q_value = target_q_values[0, policy_action].item()

        is_random_action = sample < eps_threshold
        # TODO make hard drop have a lower chance of being chosen
        # TODO have random action come from game engine and do logic there for checking if the next move is a game over
        selected_action: int = policy_action if not is_random_action else random.randrange(n_actions)
        return selected_action, policy_action, eps_threshold, (policy_q_values, double_q_value), is_random_action

    @staticmethod
    def compute_gradient_norm(model) -> float:
        total_norm: float = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5
