from typing import cast
import numpy as np
from scipy import ndimage
from gym_simpletetris.tetris.tetris_shapes import SHAPE_NAMES


def calculate_reward(board_history, done, info) -> tuple[float, dict]:
    """
    Calculate the reward and detailed statistics based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as 2D numpy arrays).
        lines_cleared_history (deque): History of lines cleared per step.
        game_over (bool): Flag indicating if the game has ended.

    Returns:
        float: Total reward.
        dict: Dictionary of detailed statistics and reward components.
    """
    current_board = board_history[-1]
    settled_board = remove_floating_blocks(current_board)

    # Calculate current board statistics
    current_stats = calculate_board_statistics(settled_board, info)

    # Calculate previous board statistics (if available)
    # TODO implement penalty if spamming the same action
    if len(board_history) > 1:
        prev_board = remove_floating_blocks(board_history[-2])
        prev_stats = calculate_board_statistics(prev_board, info.get("prev_info", {}))
    else:
        prev_stats = {}

    # Calculate rewards
    rewards = calculate_rewards(current_stats, prev_stats, info["lines_cleared_per_step"], done)

    # Combine statistics and rewards
    extra_info = {
        "current_stats": current_stats,
        "rewards": rewards,
    }

    return cast(float, rewards["total_scaled_rewards+penalties"]), extra_info


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
        "time": info.get("time", np.nan),
        "max_height": np.max(heights),
        "avg_height": np.mean(heights),
        "min_height": np.min(heights),
        "heights": heights.tolist(),
        "holes": count_holes(board),
        "bumpiness": np.sum(np.abs(np.diff(heights))),
        "density": np.sum(board) / (board.shape[0] * board.shape[1]),
        "max_height_density": np.sum(board) / max(1, (board.shape[0] * np.max(heights))),
        "lives_left": info.get("lives_left", np.nan),
        "deaths": info.get("deaths", np.nan),
        "level": info.get("level", np.nan),
        "gravity_timer": info.get("gravity_timer", 0),
        "piece_timer": info.get("piece_timer", 0),
        "gravity_interval": info.get("gravity_interval", 60),
        "anchor": info.get("anchor", (np.nan, np.nan)),
    }

    # "well_depth": calculate_well_depth(board),
    # "column_transitions": calculate_column_transitions(board),
    # "row_transitions": calculate_row_transitions(board),


def calculate_board_inputs(board, info, num_actions=20):  # TODO move this function to somewhere more relevant
    """Calculate detailed statistics for a given board state. Used for model input."""
    heights = get_column_heights(board)
    actions: list[int] = get_all_actions(info)

    padded_actions = actions[:num_actions] + [-99] * (
        num_actions - len(actions)
    )  # TODO make this the same as the SEQUENCE_LENGTH

    # Use -99 as a default value for both x and y anchors
    anchor = info.get("anchor", (-99, -99))

    current_piece = info.get("current_piece", None)
    current_piece = SHAPE_NAMES.index(current_piece) if current_piece is not None else -99

    held_piece = info.get("held_piece_name", None)
    held_piece = SHAPE_NAMES.index(held_piece) if held_piece is not None else -99

    next_pieces = info.get("next_piece", [])[:4]

    next_pieces = [SHAPE_NAMES.index(piece) for piece in next_pieces]

    padding = [-99] * (4 - len(next_pieces))

    next_pieces = next_pieces + padding

    return {
        "holes": count_holes(board),
        "bumpiness": np.sum(np.abs(np.diff(heights))),
        "lines_cleared": info.get("lines_cleared_per_step", 0),
        "x_anchor": anchor[0],
        "y_anchor": anchor[1],
        **{f"col_{i}_height": h for i, h in enumerate(heights)},
        "current_piece": current_piece,
        **{f"next_piece_{i}": piece for i, piece in enumerate(next_pieces)},
        "held_piece": held_piece,
        **{f"prev_actions_{i}": act for i, act in enumerate(padded_actions)},
    }


def calculate_rewards(
    current_stats, prev_stats, lines_cleared, game_over
) -> dict[str, dict[str, int | float] | int | float]:
    """Calculate reward components based on current and previous statistics, with scaling.

    Returns a dictionary with the following structure:

    {
        "game_over_penalty": 0,
        "lost_a_life": 0,
        "scaled_penalties": {
            "height_penalty": float,
            "hole_penalty": float,
            "max_height_diff_penalty": float,
            "hole_diff_penalty": float,
            "piece_timer": float,
        },
        "scaled_rewards": {
            "lines_cleared_reward": float,
        },
        "total_rewards": float,
        "total_penalties": float,
        "Total_Reward": float
    }

    The entries in the dictionary are:

    - game_over_penalty: a negative reward for the game being over
    - lost_a_life: a negative reward for losing a life
    - scaled_penalties: a dictionary of scaled penalties for the game
    - scaled_rewards: a dictionary of scaled rewards for the game
    - total_rewards: the total of the scaled rewards
    - total_penalties: the total of the scaled penalties
    """

    if current_stats.get("max_height", 0) < 0 or current_stats.get("holes", 0) < 0:
        raise ValueError("Invalid values in current_stats")
    if prev_stats and (prev_stats.get("max_height", 0) < 0 or prev_stats.get("holes", 0) < 0):
        raise ValueError("Invalid values in prev_stats")

    # Initialize the dictionary to ensure all keys are included
    # TODO make this into a pydantic class
    result = {
        "game_over_penalty": 0.0,
        "lost_a_life": 0.0,
        "scaled_rewards_dict": {},
        "unscaled_rewards_dict": {},
        "scaled_penalties_dict": {},
        "unscaled_penalties_dict": {},
        "total_scaled_rewards": 0.0,
        "total_unscaled_rewards": 0.0,
        "total_scaled_penalties": 0.0,
        "total_unscaled_penalties": 0.0,
        "total_scaled_rewards+penalties": 0.0,
        "total_unscaled_rewards+penalties": 0.0,
    }

    # If the game is in a new state, return only "new_game"
    if not prev_stats:
        result["new_game"] = 0
        return result

    # Determine if the player lost a life
    # TODO make this more robust and check if logic is broken elsewhere
    lost_a_life = prev_stats.get("lives_left", 0) > current_stats.get("lives_left", 0) or prev_stats.get(
        "deaths", 0
    ) < current_stats.get("deaths", 0)

    # If the game is over, apply game over penalty and return
    if game_over:
        result["game_over_penalty"] = -1
        return result

    # If a life was lost, apply the life lost penalty and return
    if lost_a_life:
        result["lost_a_life"] = -0.9
        return result

    # Start penalizing if gravity timer exceeds 30 when interval is 60
    piece_threshold = 10

    piece_timer = current_stats.get("piece_timer", 0)

    # Raw penalties and rewards
    unscaled_penalties: dict[str, float] = {
        "height_penalty": current_stats.get("max_height", 0),
        "hole_penalty": current_stats.get("holes", 0),
        # "max_height_diff_penalty": min(prev_stats.get("max_height", 0) - current_stats.get("max_height", 0), 0),
        # "hole_diff_penalty": min(prev_stats.get("holes", 0) - current_stats.get("holes", 0), 0),
        "piece_timer": piece_timer * (piece_timer >= piece_threshold),
    }

    unscaled_rewards: dict[str, float] = {
        "lines_cleared_reward": 8.0 * lines_cleared,
    }

    # Penalty boundaries (min, max)  # Assuming board is 10*20
    penalty_boundaries: dict[str, tuple[float, float]] = {
        "height_penalty": (0, 20),
        "hole_penalty": (0, 200),
        # "max_height_diff_penalty": (-10.5, 0),
        # "hole_diff_penalty": (-200, 0),
        "piece_timer": (0, piece_threshold * 2),
    }

    reward_boundaries: dict[str, tuple[float, float]] = {"lines_cleared_reward": (0, 32)}

    # Scale and normalize penalties
    scaled_penalties: dict[str, float] = {}
    for penalty_name, raw_penalty in unscaled_penalties.items():
        min_val, max_val = penalty_boundaries[penalty_name]
        normalized_penalty: float = (max_val != min_val) * ((raw_penalty - min_val) / (max_val - min_val))
        scaled_penalties[penalty_name] = abs(normalized_penalty) * -(1 / len(unscaled_penalties))

    scaled_penalties["height_penalty"] /= 2
    scaled_penalties["hole_penalty"] *= 2
    # Scale and normalize rewards
    scaled_rewards: dict[str, float] = {}
    for reward_name, raw_reward in unscaled_rewards.items():
        min_val, max_val = reward_boundaries[reward_name]
        normalized_reward: float = (max_val != min_val) * ((raw_reward - min_val) / (max_val - min_val))
        scaled_rewards[reward_name] = abs(normalized_reward) * (1 / len(unscaled_rewards))

    # Combine scaled rewards and penalties
    total_scaled_rewards: float = sum(scaled_rewards.values())
    total_scaled_penalties: float = sum(scaled_penalties.values())
    total_unscaled_rewards: float = sum(unscaled_rewards.values())
    total_unscaled_penalties: float = sum(unscaled_penalties.values())

    # Update the result with the scaled values
    result["scaled_rewards_dict"] = scaled_rewards
    result["unscaled_rewards_dict"] = unscaled_rewards
    result["scaled_penalties_dict"] = scaled_penalties
    result["unscaled_penalties_dict"] = unscaled_penalties

    result["total_scaled_rewards"] = total_scaled_rewards
    result["total_unscaled_rewards"] = total_unscaled_rewards
    result["total_scaled_penalties"] = total_scaled_penalties
    result["total_unscaled_penalties"] = total_unscaled_penalties

    result["total_scaled_rewards+penalties"] = total_scaled_rewards + total_scaled_penalties
    result["total_unscaled_rewards+penalties"] = total_unscaled_rewards + total_unscaled_penalties

    return result


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
