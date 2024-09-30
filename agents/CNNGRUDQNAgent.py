import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from models.cnn_gru import CNNGRU
from utils.replay_memory import ReplayMemory
from agents.base_agent import TetrisAgent


class CGDAgent(TetrisAgent):
    # An tetris agent that uses CNNGRU with DQN to train
    def __init__(self, board_state, action_space, config, device, model_path=None):
        self.device = device
        self.config = config
        self.n_actions = action_space.n
        self.board_state = board_state  # keep a reference to the initial state

        self.temporal_features = ["x_anchor", "y_anchor", "current_piece", "held_piece"] + [
            f"next_piece_{i}" for i in range(4)
        ]
        self.current_features = ["holes", "bumpiness", "score"] + [f"col_{i}_height" for i in range(10)]

        input_shape = (self.board_state.shape[0], self.board_state.shape[1])
        self.policy_net = CNNGRU(input_shape, self.n_actions, self.temporal_features, self.current_features).to(device)
        self.target_net = CNNGRU(input_shape, self.n_actions, self.temporal_features, self.current_features).to(device)
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

        self.board_state_deque = deque(maxlen=self.config.SEQUENCE_LENGTH)
        self.temporal_features_deque = deque(maxlen=self.config.SEQUENCE_LENGTH)
        self.reset()

    def select_action(self, state, temporal_feature, current_feature, total_steps_done):

        states_list = list(self.board_state_deque)[1:] + [state]
        states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32, device=self.device).unsqueeze(0)

        temporal_features_list = list(self.temporal_features_deque)[1:] + [temporal_feature]
        temporal_feature_tensor = torch.tensor(
            np.array(temporal_features_list), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        selected_action, policy_action, eps_threshold, q_values, is_random_action = self.select_action_static(
            (states_tensor, temporal_feature_tensor, current_feature),
            self.policy_net,
            total_steps_done,
            self.n_actions,
            self.config.EPS_START,
            self.config.EPS_END,
            self.config.EPS_DECAY,
        )
        return selected_action, (policy_action, eps_threshold, q_values, is_random_action)

    def update(self, state, action, next_state, reward, done):
        next_boards, next_temporal_features, next_feature = next_state

        self.board_state_deque.append(next_boards[-1])  # assuming this is how to get the lasted board
        self.temporal_features_deque.append(next_temporal_features[-1])
        self.memory.push(
            state,
            torch.tensor([[action]], device=self.device, dtype=torch.long),
            next_state,
            reward,
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

    def reset(self, state_simple=None):

        state_simple = state_simple or self.board_state
        self.board_state_deque.clear()
        self.board_state_deque.extend([state_simple] * self.config.SEQUENCE_LENGTH)

        self.temporal_features_deque.clear()
        temp_temp_feats = list(range(len(self.temporal_features)))
        self.temporal_features_deque.extend(temp_temp_feats * self.config.SEQUENCE_LENGTH)

    def optimize_model(self) -> tuple[float, tuple[float, float]]:
        if len(self.memory) < self.config.BATCH_SIZE:
            return np.nan, (np.nan, np.nan)  # No loss to report

        transitions = self.memory.sample(self.config.BATCH_SIZE)
        batch = list(zip(*transitions))

        # The batches for the inputs states
        states_batch = torch.cat(batch[0])

        action_batch = torch.stack(batch[1])

        next_states_batch = torch.cat(batch[2])

        reward_batch = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.bool, device=self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(states_batch).gather(1, action_batch)

        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
            non_final_mask = ~done_batch
            if non_final_mask.sum() > 0:
                next_state_values[non_final_mask] = self.target_net(next_states_batch[non_final_mask]).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Calculate gradient norm before clipping
        grad_norm_before = CGDAgent.compute_gradient_norm(self.policy_net)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.GRADIENT_CLIPPING)

        # Calculate gradient norm after clipping
        grad_norm_after = CGDAgent.compute_gradient_norm(self.policy_net)
        self.optimizer.step()

        return loss.item(), (grad_norm_before, grad_norm_after)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
