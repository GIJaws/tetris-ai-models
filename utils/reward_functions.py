import numpy as np
from scipy import ndimage


def calculate_reward(board_history, lines_cleared_history, done, info, window_size=5):
    """
    Calculate the reward and detailed statistics based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as 2D numpy arrays).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.
        window_size (int): Number of recent states to consider.

    Returns:
        float: Total reward.
        dict: Dictionary of detailed statistics and reward components.
    """
    current_board = board_history[-1]
    settled_board = remove_floating_blocks(current_board)

    # Calculate current board statistics
    current_stats = calculate_board_statistics(settled_board, info)

    # Calculate previous board statistics (if available)
    if len(board_history) > 1:
        prev_board = remove_floating_blocks(board_history[-2])
        prev_stats = calculate_board_statistics(prev_board, info.get("prev_info", {}))
    else:
        prev_stats = {}

    # Calculate lines cleared
    lines_cleared = lines_cleared_history[-1] if lines_cleared_history else 0

    action_history = get_all_actions(info)
    # Calculate rewards
    rewards = calculate_rewards(current_stats, prev_stats, lines_cleared, done, action_history)
    total_reward = sum(rewards.values())
    rewards["Total_Reward"] = sum(rewards.values())
    # Combine statistics and rewards
    detailed_info = {
        "current_stats": current_stats,
        "lines_cleared": lines_cleared,
        "rewards": rewards,
    }

    return total_reward, detailed_info


def get_all_actions(data, count=0, max_depth=10):
    """
    Recursively extract all actions from the given data dictionary, including
    nested "prev_info" dictionaries. Stops after reaching the maximum recursion
    depth (default=10) to prevent RecursionError.

    Args:
        data (dict): The dictionary to extract actions from.
        count (int, optional): The current recursion depth. Defaults to 0.
        max_depth (int, optional): The maximum recursion depth. Defaults to 10.

    Returns:
        list: A list of all actions extracted from the given data.
    """
    if count > max_depth:
        return []
    actions = []

    # Add current actions
    if "actions" in data:
        actions.extend(data["actions"])

    # Recursively check prev_info
    if "prev_info" in data and isinstance(data["prev_info"], dict):
        actions.extend(get_all_actions(data["prev_info"], count=(count + 1), max_depth=max_depth))

    return actions


def get_column_heights(board):
    non_zero_mask = board != 0
    heights = board.shape[1] - np.argmax(non_zero_mask, axis=1)
    return np.where(non_zero_mask.any(axis=1), heights, 0)


def calculate_board_statistics(board, info):
    """Calculate detailed statistics for a given board state."""
    heights = get_column_heights(board)

    return {
        "time": info.get("time", 0),
        "max_height": np.max(heights),
        "avg_height": np.mean(heights),
        "min_height": np.min(heights),
        "heights": heights.tolist(),
        "holes": count_holes(board),
        "bumpiness": np.sum(np.abs(np.diff(heights))),
        "density": np.sum(board) / (board.shape[0] * board.shape[1]),
        "max_height_density": np.sum(board) / max(1, (board.shape[0] * np.max(heights))),
        "lives_left": info.get("lives_left", 0),
        "deaths": info.get("deaths", 0),
        "level": info.get("level", 0),
        "gravity_timer": info.get("gravity_timer", 0),
        "gravity_interval": info.get("gravity_interval", 0),
        "anchor": info.get("anchor", (-99, -99)),
    }

    # "well_depth": calculate_well_depth(board),
    # "column_transitions": calculate_column_transitions(board),
    # "row_transitions": calculate_row_transitions(board),


def calculate_board_inputs(board, info):
    """Calculate detailed statistics for a given board state. Used for model input."""
    heights = get_column_heights(board)
    actions: list[int] = get_all_actions(info)

    padded_actions = actions[:4] + [-99] * (4 - len(actions))

    # Use -99 as a default value for both x and y anchors
    anchor = info.get("anchor", (-99, -99))
    return {
        "holes": count_holes(board),
        "x_anchor": anchor[0],
        "y_anchor": anchor[1],
        **{f"col_{i}_height": h for i, h in enumerate(heights)},
        **{f"prev_actions_{i}": act for i, act in enumerate(padded_actions)},
    }


def calculate_rewards_OLD(current_stats, prev_stats, lines_cleared, game_over, action_history):
    """Calculate reward components based on current and previous statistics."""

    # TODO add a penalty if the bot keeps doing the same actions in a row more then lets say three times
    # action_history value example [2, 0, 0, 6, 6, 6, 6, 1, 1, 6, 6, 3, 3, 1, 1, 1, 1, 5, 5, 3, 3, 2, 2]
    # only consider the last three actions (latest actions at the start of the list)
    # recent_actions = action_history[:4]
    # unique_recent_actions = set(recent_actions)  # get the unique actions in the last four steps
    # num_repeated_actions = len(recent_actions) - len(unique_recent_actions)
    # repeated_actions_penalty = -10 * num_repeated_actions if num_repeated_actions > 2 else 0

    if not prev_stats:
        return {"new game?": 0}

    lost_a_life = (
        prev_stats["lives_left"] > current_stats["lives_left"] or prev_stats["deaths"] < current_stats["deaths"]
    )

    return {
        # "height_penalty": -0.1 * current_stats["max_height"] if current_stats["max_height"] > 15 else 0,
        # "hole_penalty": -1 * current_stats["holes"] if not lost_a_life else 0,
        "lines_cleared_reward": 8.0 * lines_cleared,
        "game_over_penalty": -100.0 if game_over else 0,
        "lost_a_life": -10 if lost_a_life else 0,
        "max_height_penalty": (prev_stats["max_height"] - current_stats["max_height"]) if not lost_a_life else 0,
        "hole_diff_penalty": (prev_stats["holes"] - current_stats["holes"]) if not lost_a_life else 0,
        "gravity_timer": (prev_stats["gravity_timer"] - current_stats["gravity_timer"]) if not lost_a_life else 0,
        # TODO clip gravity timer between -1 and 0.1
        # "bumpiness_improvement": (
        #     max(-0.1, prev_stats["bumpiness"] - current_stats["bumpiness"]) if not lost_a_life else 0
        # ),
        # "repeated_actions_penalty": repeated_actions_penalty if not lost_a_life else 0,
    }


def calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history):
    """Calculate reward components based on current and previous statistics."""

    if not prev_stats:
        return {"new game?": 0}

    lost_a_life = (
        prev_stats["lives_left"] > current_stats["lives_left"] or prev_stats["deaths"] < current_stats["deaths"]
    )

    if game_over:
        return {"game_over_penalty": -500.0}

    if lost_a_life:
        return {"lost_a_life": -50}

    grav_timer = current_stats["gravity_timer"]

    return {
        "height_penalty": min(-0.025 * current_stats["max_height"] ** 2, 0),
        "hole_penalty": -0.025 * current_stats["holes"] ** 2,
        "lines_cleared_reward": 8.0 * lines_cleared,
        "max_height_diff": (prev_stats["max_height"] - current_stats["max_height"]),
        "hole_diff": (prev_stats["holes"] - current_stats["holes"]),
        "gravity_timer": min(-0.025 * grav_timer**2 + 5, 0),
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
