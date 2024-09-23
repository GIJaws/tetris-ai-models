import numpy as np
import itertools
from gym_simpletetris.tetris.tetris_shapes import BASIC_ACTIONS


# Reverse mapping: action name to index
ACTION_NAME_TO_INDEX = {name: action for action, name in BASIC_ACTIONS.items()}

# List of valid action combinations based on Tetris engine logic
VALID_ACTION_COMBINATIONS = [
    ["idle"],
    ["left"],
    ["right"],
    ["hard_drop"],
    ["soft_drop"],
    ["rotate_left"],
    ["rotate_right"],
    ["hold_swap"],
    ["left", "soft_drop"],
    ["right", "soft_drop"],
    ["rotate_left", "soft_drop"],
    ["rotate_right", "soft_drop"],
    # Add more valid combinations if necessary...
]


def create_action_combinations():
    """
    Generates a dictionary mapping unique integers to action combinations represented as lists of action indices.
    Returns:
        dict: Mapping from integer identifiers to lists of action indices.
    """
    action_combinations = {}
    for idx, combination in enumerate(VALID_ACTION_COMBINATIONS):
        action_indices = [ACTION_NAME_TO_INDEX[action] for action in combination]
        action_combinations[idx] = action_indices
    return action_combinations


# ACTION_COMBINATIONS = create_action_combinations()

ACTION_COMBINATIONS = {
    0: [0],  # Move Left
    1: [1],  # Move Right
    2: [2],  # Hard Drop
    3: [3],  # Soft Drop
    4: [4],  # Rotate Left
    5: [5],  # Rotate Right
    6: [6],  # Hold/Swap
    # 7: [7],  # Idle
}


def bitmask_to_actions(action_bitmask):
    """
    Converts a bitmask integer to a list of basic action indices.
    Args:
        action_bitmask (int): Bitmask representing the action combination.
    Returns:
        list: List of action indices corresponding to the bitmask.
    """
    actions = []
    for action, name in BASIC_ACTIONS.items():
        if action_bitmask & (1 << action):
            actions.append(action)
    return actions


def simplify_board(board):
    """
    Simplifies the board representation by converting it to a 2D binary array.

    Args:
        board (np.ndarray): The original board with shape (width, height, channels).

    Returns:
        np.ndarray: Simplified board with shape (width, height) as float32.
    """
    # Assuming board shape is (10, 40, 3)
    return np.any(board != 0, axis=2).astype(np.float32)


def format_value(v):
    if isinstance(v, list):
        return ", ".join([format_value(x) for x in v])
    return f"{v:.0f}" if isinstance(v, int) or v.is_integer() else f"{v:.6f}"
