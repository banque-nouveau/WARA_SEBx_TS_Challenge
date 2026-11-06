from pathlib import Path

import numpy as np
import torch

from amgm import config
from amgm.data.base import FeatureNormalizer
from amgm.models.linear_trend.runner import LinearTrendRunner


class model:
    """Baseline adapter: average the predictions of all submitted checkpoints."""

    def __init__(self):
        checkpoint_dir = (
            config.ws_dir / "linear_trend" / "logs" / "train_regression_residual"
            / "daily_252_21_US_Stocks" / "checkpoints"
        )
        self.checkpoint_files = sorted(Path(checkpoint_dir).glob("*.ckpt"))
        if not self.checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def predict(self, inputs):
        raw_features = np.asarray(inputs.features, dtype=np.float32)
        if raw_features.ndim != 3 or raw_features.shape[1:] != (252, 1):
            raise ValueError("Prediction features must have shape (N, 252, 1)")
        if len(raw_features) == 0:
            raise ValueError("Prediction inputs must not be empty")
        if not np.isfinite(raw_features).all():
            raise ValueError("Prediction features must be finite")
        normalized = FeatureNormalizer.sample_minmax(raw_features[:, :, 0])
        features = torch.as_tensor(normalized[:, :, np.newaxis])
        predictions = np.zeros(len(features), dtype=np.float64)
        for checkpoint in self.checkpoint_files:
            runner = LinearTrendRunner.load_from_checkpoint(
                checkpoint, map_location="cpu", weights_only=False
            )
            runner.eval()
            with torch.inference_mode():
                batches = [runner(batch).reshape(-1).cpu().numpy() for batch in features.split(200)]
            current = np.concatenate(batches)
            if current.shape != predictions.shape:
                raise ValueError(f"Checkpoint {checkpoint} must return one prediction per sample")
            predictions += current
        return predictions / len(self.checkpoint_files)
