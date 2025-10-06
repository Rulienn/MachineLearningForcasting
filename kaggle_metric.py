"""
Custom quantile error metric for forecasting raw material weights.

This metric expects three DataFrames:
- submission: contains predicted weight per ID
- solution: contains actual (true) weight per ID

The metric calculates quantile error over the predicted and actual weights.

Required columns:
- submission: [row_id_column_name, 'predicted_weight']
- solution: [row_id_column_name, 'weight']
"""

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    """Raise this for participant-facing errors."""

    pass


def quantile_error(actual: np.ndarray, predicted: np.ndarray, q: float = 0.2) -> float:
    """Quantile loss (pinball loss) for quantile q."""
    if np.any(actual < 0) or np.any(predicted < 0):
        raise ParticipantVisibleError("Values must be non-negative.")

    diff = actual - predicted
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "ID"
) -> float:
    """
    Compute the 0.2 quantile of underestimation error between predictions and true weights.

    Args:
        solution (pd.DataFrame): True weights per ID. Must have [row_id_column_name, 'weight']
        submission (pd.DataFrame): Predicted weights per ID. Must have [row_id_column_name, 'predicted_weight']
        row_id_column_name: Name of the column containing the ID.

    Returns:
        float: 0.2 quantile of underestimation error
    """

    for col in [row_id_column_name, "predicted_weight"]:
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission is missing column: {col}")

    if not pd.api.types.is_numeric_dtype(submission["predicted_weight"]):
        raise ParticipantVisibleError(
            "'predicted_weight' in submission must be numeric."
        )

    submission_filtered = submission[
        submission[row_id_column_name].isin(solution[row_id_column_name])
    ]

    merged = pd.merge(
        solution,
        submission_filtered,
        on=row_id_column_name,
        how="left",
        validate="one_to_one",
    )

    if merged["predicted_weight"].isnull().any():
        missing_ids = merged.loc[
            merged["predicted_weight"].isnull(), row_id_column_name
        ].tolist()
        raise ParticipantVisibleError(
            f"Missing predictions for required ID(s): {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}"
        )

    actual = merged["weight"].values
    predicted = merged["predicted_weight"].values

    try:
        result = quantile_error(actual, predicted, q=0.2)
    except Exception as e:
        raise ParticipantVisibleError(
            f"Error during underestimation quantile calculation: {e}"
        )

    if not np.isfinite(result):
        raise ParticipantVisibleError("Final quantile error is not a finite number.")

    return float(result)
