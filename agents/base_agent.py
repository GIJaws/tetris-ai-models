from abc import ABC, abstractmethod
from typing import Any


class TetrisAgent(ABC):
    @abstractmethod
    def select_action(self, state, info, total_steps_done) -> tuple[int, tuple[Any, ...]]:
        pass

    @abstractmethod
    def update(self, state_simple, action, reward, next_state_simple, done, info, next_info):
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
