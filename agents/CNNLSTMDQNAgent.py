from markdown.util import deprecated
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from models.cnn_lstm_dqn import CNNLSTMDQN
from utils.replay_memory import ReplayMemory
from agents.base_agent import TetrisAgent
from utils.reward_functions import calculate_board_inputs
from gym_simpletetris.tetris.tetris_shapes import simplify_board


@deprecated("Use CNNLSTMDQNAgent instead")
class CNNLSTMDQNAgent(TetrisAgent):
    def __init__(self, state_simple, input_shape, action_space, config, device, model_path=None):
        self.device = device
        self.config = config
        self.n_actions = action_space.n

        self.policy_net = CNNLSTMDQN(input_shape, self.n_actions, n_features=23).to(device)
        self.target_net = CNNLSTMDQN(input_shape, self.n_actions, n_features=23).to(device)
        if model_path:
            self.load_model(model_path)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=float(self.config.LEARNING_RATE),
                weight_decay=float(self.config.WEIGHT_DECAY),
            )
        self.target_net.eval()

        self.memory = ReplayMemory(self.config.MEMORY_SIZE)

        self.state_deque = deque([state_simple] * self.config.SEQUENCE_LENGTH, maxlen=self.config.SEQUENCE_LENGTH)

    def select_action_old(self, state, info, total_steps_done):
        state_simple = simplify_board(state)
        combined_state: tuple[torch.Tensor, torch.Tensor] = self._prepare_input(
            list(calculate_board_inputs(state_simple, info, num_actions=self.config.SEQUENCE_LENGTH).values())
        )

        selected_action, policy_action, eps_threshold, q_values, is_random_action = (
            CNNLSTMDQNAgent.select_action_static(
                combined_state,
                self.policy_net,
                total_steps_done,
                self.n_actions,
                self.config.EPS_START,
                self.config.EPS_END,
                self.config.EPS_DECAY,
            )
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

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=float(self.config.LEARNING_RATE),
                weight_decay=float(self.config.WEIGHT_DECAY),
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=float(self.config.LEARNING_RATE),
                weight_decay=float(self.config.WEIGHT_DECAY),
            )
        print(f"Model loaded from {path}")

    def save_model(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"Model saved to {path}")

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
        grad_norm_before = CNNLSTMDQNAgent.compute_gradient_norm(self.policy_net)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.GRADIENT_CLIPPING)

        # Calculate gradient norm after clipping
        grad_norm_after = CNNLSTMDQNAgent.compute_gradient_norm(self.policy_net)
        self.optimizer.step()

        return loss.item(), (grad_norm_before, grad_norm_after)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _prepare_input(self, board_inputs):
        state_tensor = torch.tensor(np.array(self.state_deque), dtype=torch.float32, device=self.device).unsqueeze(0)

        features_tensor = torch.tensor([list(board_inputs)], dtype=torch.float32, device=self.device)
        return state_tensor, features_tensor
