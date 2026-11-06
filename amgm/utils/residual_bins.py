import numpy as np


def residual_bins_21d(residuals):
    """Baseline US-stocks residual bins for a 21-day prediction horizon."""
    return {
        "lt20": np.where(residuals <= -20)[0],
        "n10to20": np.where((-20 < residuals) & (residuals < -10))[0],
        "n5to10": np.where((-10 <= residuals) & (residuals < -5))[0],
        "n2to5": np.where((-5 <= residuals) & (residuals < -2))[0],
        "n1to2": np.where((-2 <= residuals) & (residuals < -1))[0],
        "n0to1": np.where((-1 <= residuals) & (residuals < 0))[0],
        "zero": np.where(residuals == 0)[0],
        "0to1": np.where((0 < residuals) & (residuals <= 1))[0],
        "1to2": np.where((1 < residuals) & (residuals <= 2))[0],
        "2to5": np.where((2 < residuals) & (residuals <= 5))[0],
        "5to10": np.where((5 < residuals) & (residuals <= 10))[0],
        "10to20": np.where((10 < residuals) & (residuals < 20))[0],
        "gt20": np.where(20 <= residuals)[0],
    }
