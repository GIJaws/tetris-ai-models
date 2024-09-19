import numpy as np


def simplify_board(board):
    # Assuming board shape is (10, 40, 3)
    return np.any(board != 0, axis=2).astype(np.float32)
