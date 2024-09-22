import numpy as np
from scipy import ndimage


def calculate_reward(board_history, lines_cleared_history, game_over, time_count, window_size=5):
    """
    Calculate the reward and detailed statistics based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as 2D numpy arrays).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.
        time_count (int): Number of time steps survived.
        window_size (int): Number of recent states to consider.

    Returns:
        float: Total reward.
        dict: Dictionary of detailed statistics and reward components.
    """
    current_board = board_history[-1]
    settled_board = remove_floating_blocks(current_board)

    # Calculate current board statistics
    current_stats = calculate_board_statistics(settled_board)

    # Calculate previous board statistics (if available)
    if len(board_history) > 1:
        prev_board = remove_floating_blocks(board_history[-2])
        prev_stats = calculate_board_statistics(prev_board)
    else:
        prev_stats = current_stats.copy()

    # Calculate lines cleared
    lines_cleared = lines_cleared_history[-1] if lines_cleared_history else 0

    # Calculate rewards
    rewards = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over)
    total_reward = sum(rewards.values())

    # Combine statistics and rewards
    detailed_info = {
        "current_stats": current_stats,
        "prev_stats": prev_stats,
        "lines_cleared": lines_cleared,
        "time_survived": time_count,
        "rewards": rewards,
    }

    return total_reward, detailed_info


def get_column_heights(board):
    non_zero_mask = board != 0
    heights = board.shape[1] - np.argmax(non_zero_mask, axis=1)
    return np.where(non_zero_mask.any(axis=1), heights, 0)


def calculate_board_statistics(board):
    """Calculate detailed statistics for a given board state."""
    heights = get_column_heights(board)
    if heights.sum() > 10:
        breakpoint()

    return {
        "max_height": np.max(heights),
        "avg_height": np.mean(heights),
        "min_height": np.min(heights),
        "heights": heights.tolist(),
        "height len": len(heights),
        "holes": count_holes(board),
        "bumpiness": np.sum(np.abs(np.diff(heights))),
        "density": np.sum(board) / (board.shape[0] * board.shape[1]),
        "well_depth": calculate_well_depth(board),
        "column_transitions": calculate_column_transitions(board),
        "row_transitions": calculate_row_transitions(board),
    }


def calculate_rewards(current_stats, prev_stats, lines_cleared, game_over):
    """Calculate reward components based on current and previous statistics."""
    return {
        "height_penalty": -0.51 * current_stats["max_height"],
        "hole_penalty": -1.13 * current_stats["holes"],
        "lines_cleared_reward": 8.0 * lines_cleared,
        "game_over_penalty": -800.0 if game_over else 0,
        "height_improvement": 1 * max(0, prev_stats["max_height"] - current_stats["max_height"]),
        "hole_improvement": 1 * max(0, prev_stats["holes"] - current_stats["holes"]),
        "bumpiness_improvement": 1 * max(0, prev_stats["bumpiness"] - current_stats["bumpiness"]),
        "density_reward": 0.5 * (current_stats["density"] - prev_stats["density"]),
        "well_depth_penalty": -0.3 * current_stats["well_depth"],
        "column_transitions_penalty": -0.2 * current_stats["column_transitions"],
        "row_transitions_penalty": -0.1 * current_stats["row_transitions"],
    }


# Helper functions (implement these based on your specific needs)
def calculate_well_depth(board):
    # Calculate the depth of wells (deep gaps between columns)
    return 0  # TODO THIS IS A STUB


def calculate_column_transitions(board):
    # Calculate the number of transitions between filled and empty cells in columns
    return 0  # TODO THIS IS A STUB


def calculate_row_transitions(board):
    # Calculate the number of transitions between filled and empty cells in rows
    return 0  # TODO THIS IS A STUB


def count_holes(board):
    holes = 0
    num_cols, num_rows = board.shape
    for col in range(num_cols):
        block_found = False
        for row in range(num_rows):
            cell = board[col, row]
            if cell != 0:
                block_found = True
            elif block_found and cell == 0:
                holes += 1
    return holes


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
    # TODO Do we even need to use ndimage.label,
    #  TODO pretty sure it shouldn't matter but should check when I'm not about to go to sleep

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
