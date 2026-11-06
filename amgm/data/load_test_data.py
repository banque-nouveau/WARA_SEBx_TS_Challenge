import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from amgm.data.linear_trend import LinearTrendDataset


@dataclass(frozen=True)
class PredictionInputs:
    features: np.ndarray
    issue_ids: np.ndarray
    test_dates: np.ndarray


@dataclass(frozen=True)
class TestData:
    inputs: PredictionInputs
    targets: np.ndarray


def load_test_data(
    data_source: str | Path,
    *,
    start_date: str = "2018-12-31",
    end_date: str = "2020-12-31",
    num_iids: int | None = 500,
    seed: int = 1,
) -> TestData:
    """Prepare 252-day input windows and 21-day residual targets from US stock data."""
    data_source = Path(data_source)
    if not data_source.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_source}")
    files = [data_source / f"security_data.{extension}" for extension in ("csv", "txt")]
    files = [path for path in files if path.is_file()]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one security_data.csv or security_data.txt in {data_source}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")

    issue_ids_set = set()
    for chunk in pd.read_csv(files[0], sep="\t", usecols=["IssueId"], dtype={"IssueId": str}, chunksize=500_000):
        if chunk["IssueId"].isna().any():
            raise ValueError("security_data contains missing IssueIds")
        issue_ids_set.update(chunk["IssueId"].unique())
    issue_ids = sorted(issue_ids_set)
    if num_iids is not None:
        if num_iids <= 0 or num_iids > len(issue_ids):
            raise ValueError(f"num_iids must be between 1 and {len(issue_ids)}")
        issue_ids = sorted(random.Random(seed).sample(issue_ids, num_iids))
    if not issue_ids:
        raise ValueError("security_data contains no IssueIds")

    selected = []
    for chunk in pd.read_csv(
        files[0], sep="\t", usecols=["IssueId", "Date", "ClAdjLoc"],
        dtype={"IssueId": str}, parse_dates=["Date"], chunksize=500_000,
    ):
        mask = chunk["IssueId"].isin(issue_ids) & chunk["Date"].between(start, end)
        if mask.any():
            selected.append(chunk.loc[mask])
    if not selected:
        raise ValueError("No price records found for the selected IssueIds and date range")
    securities = pd.concat(selected, ignore_index=True).sort_values(["IssueId", "Date"])

    dataset = LinearTrendDataset(
        split="val",
        issue_ids=issue_ids,
        start_date=start_date,
        end_date_train=start_date,
        end_date_test=end_date,
        window_size_train=252,
        window_size_test=21,
        task_type="regression",
        label_variable="label_residual",
        time_bar="daily",
        feature_names=["ClAdjLoc"],
        normalization={"ClAdjLoc": "sample_minmax"},
        normalize_input_features=False,
        balancing=None,
        uploaded_df=securities,
    )
    if len(dataset.targets) == 0:
        raise ValueError("No evaluation samples were created; check dates and available history")
    if not np.isfinite(dataset.features).all() or not np.isfinite(dataset.targets).all():
        raise ValueError("Evaluation features and targets must be finite")

    return TestData(
        inputs=PredictionInputs(dataset.features, dataset.issue_ids, dataset.test_dates),
        targets=dataset.targets.reshape(-1),
    )
