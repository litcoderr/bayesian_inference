import numpy as np


def recall(pred_dist: np.array, gt: np.array, r: int = 1):
    """
    Args:
        pred_dist: [N, M]
        gt: [N]
        r: recall value
    
    Returns:
        recall percentage
    """
    indices = np.argsort(-pred_dist)
    top_r_indices = indices[:, :r]

    is_in_top_r = np.any(top_r_indices == gt[:, np.newaxis], axis=1)

    return np.mean(is_in_top_r) * 100
