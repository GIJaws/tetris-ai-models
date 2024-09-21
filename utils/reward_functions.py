# tetris-ai-models\utils\reward_functions.py
import numpy as np
from scipy import ndimage


def calculate_reward(board_history, lines_cleared_history, game_over, time_count, window_size=5):
    """
    Calculate the reward based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as 2D numpy arrays).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.
        time_count (int): Number of time steps survived.
        window_size (int): Number of recent states to consider (not used in this version).

    Returns:
        float: Total reward.
        dict: Dictionary of individual reward components.
    """
    # Get the current board state (most recent in history)
    current_board = board_history[-1]

    # Remove floating blocks (including the current falling piece)
    settled_board = remove_floating_blocks(current_board)

    # Calculate basic metrics
    heights = np.sum(settled_board, axis=0)
    max_height = np.max(heights)
    bumpiness = np.sum(np.abs(np.diff(heights)))
    holes = np.sum((settled_board == 0) & (np.cumsum(settled_board, axis=0) > 0))

    # Lines cleared in the most recent move
    lines_cleared = lines_cleared_history[-1] if lines_cleared_history else 0

    # Reward components
    height_penalty = -0.51 * max_height
    hole_penalty = -1.13 * holes
    bumpiness_penalty = -0.18 * bumpiness
    lines_cleared_reward = 8.0 * lines_cleared
    survival_reward = 0.01 * time_count  # Small reward for surviving each step

    # Game over penalty
    game_over_penalty = -8.0 if game_over else 0

    # Calculate total reward
    total_reward = (
        height_penalty
        + hole_penalty
        # + bumpiness_penalty
        + lines_cleared_reward
        + game_over_penalty
        # survival_reward +
    )

    # Collect reward components in a dictionary
    reward_components = {
        "height_penalty": height_penalty,
        "hole_penalty": hole_penalty,
        "bumpiness_penalty": bumpiness_penalty,
        "lines_cleared_reward": lines_cleared_reward,
        "survival_reward": survival_reward,
        "game_over_penalty": game_over_penalty,
        "total_reward": total_reward,
    }

    return total_reward, reward_components, (current_board, settled_board)


def calculate_reward_OLD(board_history, lines_cleared_history, game_over, time_count, window_size=5):
    """
    Calculate the reward based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as 2D numpy arrays).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.
        time_count (int): Number of time steps survived.
        window_size (int): Number of recent states to consider for rolling comparison.

    Returns:
        float: Total reward.
        dict: Dictionary of individual reward components.
    """
    # Initialize reward components
    survival_reward = 0
    lines_cleared_reward = 0
    holes_reward = 0
    max_height_reward = 0
    bumpiness_reward = 0
    game_over_penalty = 0
    avg_height = 0
    # 1. Survival Reward
    survival_reward = min(100, time_count * 0.01)  # Capped to prevent excessive rewards

    # 2. Reward for Lines Cleared
    # Calculate total lines cleared in the current step
    total_lines_cleared = sum(lines_cleared_history)
    lines_cleared_reward = total_lines_cleared * 100  # Base reward per line
    if total_lines_cleared > 1:
        lines_cleared_reward += (total_lines_cleared - 1) * 50  # Bonus for multiple lines

    # Weights for the metrics
    weights = {
        "holes": -1,  # Penalize increase in holes
        "max_height": -0.10,  # Penalize increase in max height
        "bumpiness": -0.10,  # Penalize increase in bumpiness
        "avg_height": 1,
    }

    # Decay rate for older states
    decay_rate = 0.1  # Adjust between 0 and 1; lower means faster decay

    board_without_piece_history = [remove_floating_blocks(board) for board in board_history]

    # 3. Rolling Board Comparison with Decaying Weights
    recent_boards = list(board_without_piece_history)[-window_size:]
    cumulative_metrics_diff = {metric: 0.0 for metric in weights}

    n = len(recent_boards)
    for i in range(1, n):
        prev_board = recent_boards[i - 1]
        current_board = recent_boards[i]

        prev_metrics = calculate_board_metrics(prev_board)
        current_metrics = calculate_board_metrics(current_board)

        # Calculate differences
        diff = {metric: current_metrics[metric] - prev_metrics[metric] for metric in weights.keys()}

        # Apply decay weight (more recent differences have higher weight)
        decay_weight = decay_rate ** (i - 1)

        # Accumulate weighted differences
        for metric in weights.keys():
            cumulative_metrics_diff[metric] += decay_weight * diff[metric]

    # Apply penalties or rewards per metric with caps
    max_component_reward = 50  # Adjust as needed to prevent stacking
    for metric, total_diff in cumulative_metrics_diff.items():
        if total_diff < 0:
            # Improvement: decrease in metric
            component_reward = abs(total_diff) * abs(weights[metric]) * 0.5  # Reward half the improvement
        elif total_diff > 0:
            # Deterioration: increase in metric
            component_reward = total_diff * weights[metric]  # Apply penalty
        else:
            component_reward = 0

        # Cap the contribution from this component
        component_reward = np.clip(component_reward, -max_component_reward, max_component_reward)

        # Add to component rewards
        if metric == "holes":
            holes_reward += component_reward
        elif metric == "max_height":
            max_height_reward += component_reward
        elif metric == "bumpiness":
            bumpiness_reward += component_reward
        elif metric == "avg_height":
            avg_height += component_reward

    # 4. Penalty for Game Over
    if game_over:
        game_over_penalty = float("-inf")  # TODO is -inf to much??

    # Total reward
    total_reward = (
        survival_reward
        + lines_cleared_reward
        # + holes_reward
        + max_height_reward
        + bumpiness_reward
        + game_over_penalty
    )

    # Collect reward components in a dictionary
    reward_components = {
        "survival_reward": survival_reward,
        "lines_cleared_reward": lines_cleared_reward,
        "holes_reward": holes_reward,
        "max_height_reward": max_height_reward,
        "avg_height": avg_height,
        "bumpiness_reward": bumpiness_reward,
        # "game_over_penalty": game_over_penalty,
        "total_reward": total_reward,
    }

    return total_reward, reward_components, (board_history[-1], board_without_piece_history[-1])


def remove_floating_blocks(board):
    """
    Removes floating blocks (including the falling piece) from the board.

    Args:
        board (np.ndarray): The simplified board (binary, 0 for empty, 1 for blocks).

    Returns:
        np.ndarray: Board with only settled pieces.
    """
    # Label connected components
    labeled_board, num_features = ndimage.label(board)

    # Create a mask for settled pieces connected to the bottom
    connected_to_bottom = np.zeros_like(board, dtype=bool)

    # Check bottom row for pieces
    bottom_labels = set(labeled_board[:, -1]) - {0}

    # Mark all blocks with labels found in the bottom row
    for label in bottom_labels:
        connected_to_bottom |= labeled_board == label

    # Use the mask to keep only settled blocks and remove floating ones
    settled_board = board * connected_to_bottom

    return settled_board


def calculate_board_metrics(board):
    """
    Calculate key metrics from the board state.

    Args:
        board (np.ndarray): Simplified board with shape (height, width).

    Returns:
        dict: Metrics including max height`, number of holes, and bumpiness.
    """
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])
    max_height = np.max(heights)
    avg_height = np.mean(heights)
    holes = sum(
        np.sum(~board[np.argmax(board[:, x] != 0) :, x].astype(bool))
        for x in range(board.shape[1])
        if np.any(board[:, x])
    )
    bumpiness = np.sum(np.abs(np.diff(heights)))

    return {"max_height": max_height, "holes": holes, "bumpiness": bumpiness, "avg_height": avg_height}
