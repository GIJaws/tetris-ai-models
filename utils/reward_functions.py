import numpy as np


def calculate_reward(board_history, lines_cleared_history, game_over, time_count, window_size=5):
    """
    Calculate the reward based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as a 2D numpy array).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.
        time_count (int): Number of time steps survived.
        window_size (int): Number of recent states to consider for rolling comparison.

    Returns:
        float: Calculated reward.
    """
    reward = 0

    # 1. Survival Reward
    survival_reward = min(100, time_count * 0.1)  # Capped to prevent excessive rewards
    reward += survival_reward

    # 2. Reward for Lines Cleared
    # Calculate total lines cleared in the current step
    total_lines_cleared = sum(lines_cleared_history)
    reward += total_lines_cleared * 100  # Base reward per line
    if total_lines_cleared > 1:
        reward += (total_lines_cleared - 1) * 50  # Bonus for multiple lines

    # Define weights (these can be tuned)
    weights = {
        "holes": -10.0,  # Penalize increase in holes
        "max_height": -5.0,  # Penalize increase in max height
        "bumpiness": -1.0,  # Penalize increase in bumpiness
    }

    # 3. Rolling Board Comparison
    recent_boards = list(board_history)[-window_size:]
    cumulative_metrics_diff = {"holes": 0, "max_height": 0, "bumpiness": 0}

    # Iterate through the window to accumulate differences
    for i in range(1, min(window_size, len(recent_boards))):
        prev_board = recent_boards[i - 1]
        current_board = recent_boards[i]

        prev_metrics = calculate_board_metrics(prev_board)
        current_metrics = calculate_board_metrics(current_board)

        # Calculate differences
        cumulative_metrics_diff["holes"] += current_metrics["holes"] - prev_metrics["holes"]
        cumulative_metrics_diff["bumpiness"] += current_metrics["bumpiness"] - prev_metrics["bumpiness"]

        # Penalize only if height exceeds a certain threshold (e.g., near the top)
        if current_metrics["max_height"] > 15:  # Assuming board height is 20
            cumulative_metrics_diff["max_height"] += current_metrics["max_height"] - prev_metrics["max_height"]

    # Apply penalties or rewards
    for metric, diff in cumulative_metrics_diff.items():
        if diff < 0:
            # Improvement: decrease in metric
            reward += abs(diff) * abs(weights[metric]) * 0.5  # Reward half the improvement
        elif diff > 0:
            # Deterioration: increase in metric
            reward += diff * weights[metric]  # Apply penalty

    # 5. Penalty for Game Over
    if game_over:
        reward -= 500  # Significant penalty for losing the game

    return reward


def calculate_board_metrics(board):
    """
    Calculate key metrics from the board state.

    Args:
        board (np.ndarray): Simplified board with shape (width, height).

    Returns:
        dict: Metrics including max height, number of holes, and bumpiness.
    """
    heights = np.array([board.shape[0] - np.argmax(column) if np.any(column) else 0 for column in board.T])
    max_height = np.max(heights)
    holes = sum(np.sum(~board[:, x].astype(bool)[np.argmax(board[:, x] != 0) :]) for x in range(board.shape[1]))
    bumpiness = np.sum(np.abs(np.diff(heights)))

    return {"max_height": max_height, "holes": holes, "bumpiness": bumpiness}
