import numpy as np

from amgm.utils.common import metric_regression_residual
from amgm.utils.residual_bins import residual_bins_21d


def calculate_metrics(targets, predictions) -> dict:
    """Calculate single-run residual metrics using the baseline's 21-day bins."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if targets.ndim != 1 or predictions.ndim != 1 or targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have matching shapes (N,) or (N, 1)")
    if not targets.size:
        raise ValueError("Targets and predictions must not be empty")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Targets and predictions must be finite")

    mse, bin_maes, bin_names = metric_regression_residual(
        targets, predictions, window_size_test=21,
        bins_fn=residual_bins_21d,
    )
    populated_maes = [float(value) for value in bin_maes if not np.isnan(value)]
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(targets - predictions))),
        "mae_per_bin": {
            name: None if np.isnan(value) else float(value)
            for name, value in zip(bin_names, bin_maes)
        },
        "average_mae_across_bins": float(np.mean(populated_maes)),
        "num_samples": int(targets.size),
    }
