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
    current_board = board_history[-1]
    settled_board = remove_floating_blocks(current_board)

    # Calculate metrics
    heights = np.sum(settled_board, axis=0)
    max_height = np.max(heights)
    bumpiness = np.sum(np.abs(np.diff(heights)))
    holes = count_holes(settled_board)

    # Previous board metrics (if available)
    if len(board_history) > 1:
        prev_board = remove_floating_blocks(board_history[-2])
        prev_holes = count_holes(prev_board)
        prev_max_height = np.max(np.sum(prev_board, axis=0))
        prev_bumpiness = np.sum(np.abs(np.diff(np.sum(prev_board, axis=0))))
    else:
        prev_holes = holes
        prev_max_height = max_height
        prev_bumpiness = bumpiness

    lines_cleared = lines_cleared_history[-1] if lines_cleared_history else 0

    # Reward components
    height_penalty = -0.51 * max_height
    hole_penalty = -1 * holes
    lines_cleared_reward = 8.0 * lines_cleared
    game_over_penalty = -8.0 if game_over else 0

    # Improvement rewards
    height_improvement = 0.25 * max(0, prev_max_height - max_height)
    hole_improvement = 0.5 * max(0, prev_holes - holes)
    bumpiness_improvement = 0.1 * max(0, prev_bumpiness - bumpiness)

    total_reward = (
        height_penalty
        + hole_penalty
        + lines_cleared_reward
        + game_over_penalty
        + height_improvement
        + hole_improvement
        + bumpiness_improvement
    )

    reward_components = {
        "height_penalty": height_penalty,
        "hole_penalty": hole_penalty,
        "lines_cleared_reward": lines_cleared_reward,
        "game_over_penalty": game_over_penalty,
        "height_improvement": height_improvement,
        "hole_improvement": hole_improvement,
        "bumpiness_improvement": bumpiness_improvement,
        "total_reward": total_reward,
    }

    return total_reward, reward_components, (current_board, settled_board)


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
