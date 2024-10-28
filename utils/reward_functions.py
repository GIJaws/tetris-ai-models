from typing import cast
from markdown.util import deprecated
import numpy as np
from scipy import ndimage

from gym_simpletetris.core.tetris_engine import GameState
from gym_simpletetris.core.pieces import PieceType
from gym_simpletetris.core.game_actions import GameAction


def calculate_reward(board_history, done, info) -> tuple[float, dict]:
    """
    Calculate the reward and detailed statistics based on the history of board states and lines cleared.

    Args:
        board_history (deque): History of board states (each as 2D numpy arrays).

    Returns:
        float: Total reward.
        dict: Dictionary of detailed statistics and reward components.
    """
    settled_board = info["game_state"].board.grid

    # Calculate current board statistics
    current_stats = calculate_board_statistics(settled_board, info)

    # Calculate previous board statistics (if available)
    # TODO implement penalty if spamming the same action or bad finness
    # if len(board_history) > 1:
    #     prev_stats = calculate_board_statistics(
    #         info["prev_info"].get("settled_board", settled_board), info["prev_info"]
    #     )
    # else:
    #     prev_stats = {
    #         "lives_left": 0,
    #         "deaths": 1,
    #         "hold_used": False,
    #     }

    # held_penalty = info["hold_used"] and prev_stats["hold_used"]

    # Calculate rewards
    # rewards = calculate_rewards(current_stats, prev_stats, info["lines_cleared_per_step"], done, held_penalty, info)

    # Combine statistics and rewards
    extra_info = {
        "current_stats": current_stats,
        # "rewards": rewards,
    }

    return info["score"], extra_info


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


def calculate_board_statistics(board, info):
    """Calculate detailed statistics for a given board state."""

    return {
        "time": info.get("time", 0),
        "random_valid_move_str": info.get("random_valid_move_str", "null"),
        # "max_height": np.max(info["heights"]),
        "avg_height": np.mean(info["heights"]),
        # "min_height": np.min(info["heights"]),
        "heights": info["heights"],
        "bumpiness": info["bumpiness"],
        "density": np.sum(board) / (board.shape[0] * board.shape[1]),
        "max_height_density": np.sum(board) / max(1, (board.shape[0] * np.max(info["heights"]))),
        "lives_left": info.get("lives_left", -1),
        "deaths": info.get("deaths", 1),
        # "level": info.get("level", 1),
        "gravity_timer": info.get("gravity_timer", 0),
        "piece_timer": info.get("piece_timer", 0),
        "gravity_interval": info.get("gravity_interval", 60),
        # "anchor": info.get("anchor", (np.nan, np.nan)),
        # "statistics": info["statistics"],
        "hold_used": info["hold_used"],
        "current_piece_coords": info["current_piece_coords"],
        "ghost_piece_coords": info["ghost_piece_coords"],
        "is_current_finesse": info["is_current_finesse"],
        "is_finesse_complete": info["is_finesse_complete"],
        "score": info["score"],
        "held_piece_name": info["held_piece_name"],
        "holes": info["holes"],
        "current_piece": info["current_piece"],
    }


def calculate_rewards(
    current_stats, prev_stats, lines_cleared, game_over, held_penalty: bool, info
) -> dict[str, dict[str, int | float] | int | float]:
    """Calculate reward components based on current and previous statistics, with scaling.

    TODO Returns a dictionary with the following structure: TODO

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

    # Determine if the player lost a life
    # TODO make this more robust and check if logic is broken elsewhere

    # piece_threshold = 20

    # max_height = current_stats.get("max_height", 0)
    cur_time = info.get("time", 1)
    # Raw penalties and rewards
    unscaled_penalties: dict[str, float] = {
        # "height_penalty": max_height * (max_height >= 17),
        "hole_penalty": info["holes"],
        # "piece_timer": info["piece_timer"] * (info["piece_timer"] >= piece_threshold),
        "is_not_finesse": not info["is_current_finesse"],
        "held_penalty": held_penalty,
        "hole_increase": info["holes"] > info["old_holes"],
        "time_penalty": 1 / max(cur_time - (180), 1),
    }

    # Penalty boundaries (min, max)  # Assuming board is 10*20
    penalty_boundaries: dict[str, tuple[float, float]] = {
        # "height_penalty": (0, 20),
        # "piece_timer": (0, piece_threshold * 10),
        "hole_penalty": (0, 300),  # ? actually max is 200 but scaling with this
        "is_not_finesse": (0, 12),
        "held_penalty": (0, 12),  # actually (0, 1) but this makes it 1/16 instead of 1 when true
        "hole_increase": (0, 12),
        "time_penalty": (0, 6),
    }

    unscaled_rewards: dict[str, float] = {
        "lines_cleared_per_step": 8.0 * lines_cleared,
        "hole_decrease": info["holes"] < info["old_holes"],
        # "piece_place": info["piece_timer"] == 0 and cur_time > 1,
    }
    reward_boundaries: dict[str, tuple[float, float]] = {
        "lines_cleared_per_step": (0, 32),
        "hole_decrease": (0, 24),
        # "piece_place": (0, 24),
    }

    # Scale and normalize penalties
    scaled_penalties: dict[str, float] = {}
    for penalty_name, raw_penalty in unscaled_penalties.items():
        min_val, max_val = penalty_boundaries[penalty_name]
        normalized_penalty: float = (max_val != min_val) * ((raw_penalty - min_val) / (max_val - min_val))
        # scaled_penalties[penalty_name] = abs(normalized_penalty) * -(1 / len(unscaled_penalties))
        scaled_penalties[penalty_name] = abs(normalized_penalty) * -1  # not normalising between penatlies

    # scaled_penalties["height_penalty"] /= 3
    # scaled_penalties["hole_penalty"] *= 3

    # Scale and normalize rewards
    scaled_rewards: dict[str, float] = {}
    for reward_name, raw_reward in unscaled_rewards.items():
        min_val, max_val = reward_boundaries[reward_name]
        normalized_reward: float = (max_val != min_val) * ((raw_reward - min_val) / (max_val - min_val))
        scaled_rewards[reward_name] = abs(normalized_reward) * (1 / len(unscaled_rewards))

    # If the game is over, apply game over penalty
    if game_over:
        scaled_penalties["game_over_penalty"] = -1
        unscaled_penalties["game_over_penalty"] = -1

    # if info["lost_a_life"]:
    #     scaled_penalties["lost_a_life"] = -0.9
    #     unscaled_penalties["lost_a_life"] = -0.9

    death = info["lost_a_life"] or game_over or info["game_over"]

    if death:
        if cur_time <= 100:
            max_penalty = -10
            if game_over:
                scaled_penalties["game_over_penalty"] = -10
                unscaled_penalties["game_over_penalty"] = -10
        else:
            max_penalty = -1
            if game_over:
                scaled_penalties["game_over_penalty"] = -1
                unscaled_penalties["game_over_penalty"] = -1
    else:
        max_penalty = -0.8

    max_penalty = (-100 if cur_time <= 60 else -1) if death else -0.8
    # Combine scaled rewards and penalties
    total_scaled_rewards: float = sum(scaled_rewards.values())
    total_scaled_penalties: float = max(sum(scaled_penalties.values()), max_penalty)
    total_unscaled_rewards: float = sum(unscaled_rewards.values())
    total_unscaled_penalties: float = abs(sum(unscaled_penalties.values())) * -1

    # Update the result with the scaled values
    result["scaled_rewards_dict"] = scaled_rewards
    result["unscaled_rewards_dict"] = unscaled_rewards
    result["scaled_penalties_dict"] = scaled_penalties
    result["unscaled_penalties_dict"] = unscaled_penalties

    result["total_scaled_rewards"] = total_scaled_rewards
    result["total_unscaled_rewards"] = total_unscaled_rewards
    result["total_scaled_penalties"] = total_scaled_penalties
    result["total_unscaled_penalties"] = total_unscaled_penalties

    result["total_scaled_rewards+penalties"] = (
        total_scaled_rewards + total_scaled_penalties if not death else total_scaled_penalties
    )
    result["total_unscaled_rewards+penalties"] = total_unscaled_rewards + total_unscaled_penalties

    return result


# Helper functions (implement these based on your specific needs)
def calculate_well_depth(board):
    # Calculate the depth of wells (deep gaps between columns)
    return 0  # TODO THIS IS A STUB


@deprecated("this should already be in info")
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


def extract_temporal_feature(info):
    """
    Extract temporal features from the GameState.

    :return: Dict of temporal feature values
    """

    game_state = cast(GameState, info["game_state"])
    held_piece = game_state.held_piece
    held_piece_index = PieceType.piece_names().index(held_piece.name) if held_piece else -99

    num_next_pieces = 1
    next_pieces = [piece.name for piece in game_state.next_pieces[:num_next_pieces]]
    next_pieces_indices = [PieceType.piece_names().index(piece) for piece in next_pieces]
    next_pieces_indices += [-99] * (num_next_pieces - len(next_pieces_indices))

    current_piece = game_state.current_piece

    return {
        **{f"action_{action.name}": game_state.info.get("actions", [None])[0] == action.name for action in GameAction},
        **{f"current_piece_{name}": current_piece.name == name for name in PieceType.piece_names()},
        **{
            f"next_piece_{i}_{name}": piece == PieceType.piece_names().index(name)
            for i, piece in enumerate(next_pieces_indices)
            for name in PieceType.piece_names()
        },
        **{
            f"held_piece_{name}": held_piece_index == PieceType.piece_names().index(name)
            for name in PieceType.piece_names()
        },
        "hold_used": game_state.hold_used,
        "time": 1 / max(game_state.current_time, 1),
    }


def extract_current_feature(info):
    """
    Extract current features from the simplified board and game info.

    :param info: Dictionary containing game state information
    :return: Dict of current feature values
    """
    game_state = cast(GameState, info["game_state"])
    return {
        "holes": game_state.board.count_holes() / 200,
        "agg_height": np.sum(game_state.board.get_column_heights()) / 200,
        # "hold_used": info["hold_used"],
        # "time": 1 / max(info.get("time", 1), 1),
        # "bumpiness": np.sum(np.abs(np.diff(info["heights"]))) / 200,
        # "score": info.get("score", 0),
        # **{f"ghost_piece_x_coords_{i}": x / 10 for i, (x, y) in enumerate(info["ghost_piece_coords"])},
        # **{f"ghost_piece_y_coords_{i}": y / 20 for i, (x, y) in enumerate(info["ghost_piece_coords"])},
        # **{f"col_{i}_height": h for i, h in enumerate(heights)},
    }
