import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from models.cnn_lstm import CNNLSTM
from utils.replay_memory import NStepPrioritizedReplayMemory
from agents.base_agent import TetrisAgent


class CLDDAgent(TetrisAgent):
    # An tetris agent that uses CNNLSTM with DQN to train
    def __init__(
        self, board_state, action_space, temporal_features, current_features, config, device, model_path=None
    ):
        self.device = device
        self.config = config
        self.n_actions = action_space.n
        self.board_state = board_state  # keep a reference to the initial state

        self.temporal_features = temporal_features
        self.current_features = current_features

        input_shape = (self.board_state.shape[0], self.board_state.shape[1])
        self.policy_net = CNNLSTM(input_shape, self.n_actions, self.temporal_features, self.current_features).to(
            device
        )
        self.target_net = CNNLSTM(input_shape, self.n_actions, self.temporal_features, self.current_features).to(
            device
        )
        if model_path:
            self.load_model(model_path)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=float(self.config.LEARNING_RATE),
                weight_decay=float(self.config.WEIGHT_DECAY),
            )
        # Set target network to eval mode and freeze its parameters
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False
        print("Policy net requires_grad:", all(p.requires_grad for p in self.policy_net.parameters()))
        print("Target net requires_grad:", all(p.requires_grad for p in self.target_net.parameters()))
        print("Policy net training:", self.policy_net.training)
        print("Target net training:", self.target_net.training)

        self.memory = NStepPrioritizedReplayMemory(
            capacity=self.config.MEMORY_SIZE,
            n_step=self.config.N_STEP,
            gamma=self.config.GAMMA,
            alpha=self.config.PRIORITY_ALPHA,
            beta_start=self.config.PRIORITY_BETA_START,
            beta_end=self.config.PRIORITY_BETA_END,
            beta_annealing_steps=self.config.PRIORITY_BETA_ANNEALING_STEPS,
            epsilon=self.config.PRIORITY_EPSILON,
        )

        self.board_state_deque = deque(maxlen=self.config.SEQUENCE_LENGTH)
        self.temporal_features_deque = deque(maxlen=self.config.SEQUENCE_LENGTH)
        self.reset()

    def select_action(self, state, temporal_feature, current_feature, total_steps_done):
        # Process board states
        states_list = np.array(list(self.board_state_deque)[1:] + [state])
        states_tensor = torch.tensor(states_list, dtype=torch.float32, device=self.device)

        # Process temporal features
        temporal_features_list = np.array(list(self.temporal_features_deque)[1:] + [temporal_feature])
        temporal_feature_tensor = torch.tensor(temporal_features_list, dtype=torch.float32, device=self.device)

        # Process current feature
        current_feature_tensor = torch.tensor(np.array([current_feature]), dtype=torch.float32, device=self.device)

        selected_action, policy_action, eps_threshold, (q_values, double_q_value), is_random_action = (
            self.select_action_static(
                (states_tensor, temporal_feature_tensor, current_feature_tensor),
                self.policy_net,
                self.target_net,
                total_steps_done,
                self.n_actions,
                self.config.EPS_START,
                self.config.EPS_END,
                self.config.EPS_DECAY,
            )
        )
        return selected_action, (policy_action, eps_threshold, (q_values, double_q_value), is_random_action)

    def update(self, state, action, next_state, reward, done, lost_a_life=False):
        _, _, feature = state
        next_board, next_temporal_feature, next_feature = next_state

        og_board_state_deque = np.array(list(self.board_state_deque))
        og_temporal_features_deque = np.array(list(self.temporal_features_deque))

        next_temporal_feature = np.array(next_temporal_feature)

        self.board_state_deque.append(next_board)
        self.temporal_features_deque.append(next_temporal_feature)

        next_board_deque = np.array(self.board_state_deque)
        next_temporal_features = np.array(self.temporal_features_deque)

        # Calculate TD error
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(og_board_state_deque).unsqueeze(0).to(self.device),
                torch.FloatTensor(og_temporal_features_deque).unsqueeze(0).to(self.device),
                torch.FloatTensor(feature).unsqueeze(0).to(self.device),
            )
            next_state_tensor = (
                torch.FloatTensor(next_board_deque).unsqueeze(0).to(self.device),
                torch.FloatTensor(next_temporal_features).unsqueeze(0).to(self.device),
                torch.FloatTensor(next_feature).unsqueeze(0).to(self.device),
            )

            current_q = self.policy_net(state_tensor)[0, action].item()
            next_q = self.target_net(next_state_tensor).max(1)[0].item()

            expected_q = reward + (self.config.GAMMA * next_q * (1 - done))
            td_error = abs(current_q - expected_q)

        self.memory.push(
            (og_board_state_deque, og_temporal_features_deque, feature),
            torch.tensor([action], device=self.device, dtype=torch.long),
            (next_board_deque, next_temporal_features, next_feature),
            reward,
            done or lost_a_life,
            td_error,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=float(self.config.LEARNING_RATE),
            weight_decay=float(self.config.WEIGHT_DECAY),
        )
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

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
        if state_simple is not None:  # TODO is this even useful?
            self.board_state = np.array(state_simple).reshape(10, 21)

        # Initialize with empty states
        empty_state = np.zeros_like(self.board_state)
        self.board_state_deque.clear()
        self.board_state_deque.extend(np.array([empty_state] * self.config.SEQUENCE_LENGTH))

        # Initialize temporal features
        empty_temporal_features = np.zeros(len(self.temporal_features)) - 99
        self.temporal_features_deque.clear()
        self.temporal_features_deque.extend(np.array([empty_temporal_features] * self.config.SEQUENCE_LENGTH))

    def optimize_model(self) -> tuple[float, tuple[float, float]]:
        if len(self.memory) < self.config.BATCH_SIZE:
            return np.nan, (np.nan, np.nan)  # No loss to report

        transitions, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
        batch = list(zip(*transitions))

        # Unpack the state tuples
        state_boards, state_temporal_features, state_current_features = zip(*batch[0])

        # Unpack the next state tuples
        next_state_boards, next_state_temporal_features, next_state_current_features = zip(*batch[2])

        # Convert to tensors
        state_boards_batch = torch.stack([torch.from_numpy(item) for item in state_boards]).to(self.device)
        state_temporal_features_batch = torch.stack([torch.from_numpy(item) for item in state_temporal_features]).to(
            self.device
        )
        state_current_features_batch = torch.stack([torch.from_numpy(item) for item in state_current_features]).to(
            self.device
        )

        next_state_boards_batch = torch.stack([torch.from_numpy(item) for item in next_state_boards]).to(self.device)
        next_state_temporal_features_batch = torch.stack(
            [torch.from_numpy(item) for item in next_state_temporal_features]
        ).to(self.device)
        next_state_current_features_batch = torch.stack(
            [torch.from_numpy(item) for item in next_state_current_features]
        ).to(self.device)

        action_batch = torch.stack(batch[1]).to(self.device)
        reward_batch = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.bool, device=self.device)

        # Convert weights to tensor
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(
            (state_boards_batch, state_temporal_features_batch, state_current_features_batch)
        ).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states using Double DQN
        with torch.no_grad():
            # Use policy net to select action
            next_state_actions = (
                self.policy_net(
                    (next_state_boards_batch, next_state_temporal_features_batch, next_state_current_features_batch)
                )
                .max(1)[1]
                .unsqueeze(1)
            )

            # Use target net to evaluate the action
            next_state_values = self.target_net(
                (next_state_boards_batch, next_state_temporal_features_batch, next_state_current_features_batch)
            ).gather(1, next_state_actions)

            next_state_values[done_batch] = 0.0

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch.unsqueeze(1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction="none")

        # Weight the loss by importance sampling weights
        weighted_loss = (loss * weights.unsqueeze(1)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()

        # Calculate gradient norm before clipping
        grad_norm_before = self.compute_gradient_norm(self.policy_net)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.GRADIENT_CLIPPING)

        # Calculate gradient norm after clipping
        grad_norm_after = self.compute_gradient_norm(self.policy_net)

        self.optimizer.step()

        # Update priorities in the replay memory
        td_errors = (state_action_values - expected_state_action_values).detach().abs().squeeze(1).cpu().numpy()

        self.memory.update_priorities(indices, td_errors)

        return weighted_loss.item(), (grad_norm_before, grad_norm_after)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target_network(self, tau=0.005):
        """
        Perform a soft update of the target network parameters.

        The update is done as follows:
        θ_target = τ * θ_policy + (1 - τ) * θ_target

        where θ_target are the target network parameters and θ_policy are the policy network parameters.

        Args:
        tau (float): The interpolation parameter. Default is 0.005.
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
