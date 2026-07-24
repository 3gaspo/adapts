"""Chronological model-local resplitting of the pooled T1+T2 payload."""

from __future__ import annotations

from typing import Any

import numpy as np


def chronological_date_slices(
    n_dates: int,
    validation_fraction: float,
) -> tuple[slice, slice]:
    """Return nonempty chronological T1 and T2 slices along the date axis."""
    n_dates = int(n_dates)
    fraction = float(validation_fraction)
    if n_dates < 2:
        raise ValueError("T1+T2 must contain at least two query dates")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    n_valid = min(n_dates - 1, max(1, int(round(fraction * n_dates))))
    boundary = n_dates - n_valid
    return slice(0, boundary), slice(boundary, n_dates)


def chronological_resplit_arrays(
    arrays: dict[str, np.ndarray],
    validation_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Split flattened arrays by whole query dates, never by user rows."""
    query_t = np.asarray(arrays["query_t"])
    if query_t.ndim != 1:
        raise ValueError("query_t must be a flattened one-dimensional array")
    unique_dates = np.unique(query_t)
    train_slice, valid_slice = chronological_date_slices(
        len(unique_dates),
        validation_fraction,
    )
    train_dates = unique_dates[train_slice]
    valid_dates = unique_dates[valid_slice]
    boundary = valid_dates[0]
    train_mask = query_t < boundary
    valid_mask = query_t >= boundary
    n_samples = query_t.shape[0]
    for name, value in arrays.items():
        if value.shape[0] != n_samples:
            raise ValueError(f"array {name!r} is not aligned with query_t")
    train = {name: value[train_mask] for name, value in arrays.items()}
    valid = {name: value[valid_mask] for name, value in arrays.items()}
    metadata = {
        "validation_fraction": float(validation_fraction),
        "available_dates": int(len(unique_dates)),
        "t1_dates": int(len(train_dates)),
        "t2_dates": int(len(valid_dates)),
        "t1_samples": int(train_mask.sum()),
        "t2_samples": int(valid_mask.sum()),
        "t1_first_query_date": int(train_dates[0]),
        "t1_last_query_date": int(train_dates[-1]),
        "t2_first_query_date": int(valid_dates[0]),
        "t2_last_query_date": int(valid_dates[-1]),
    }
    return train, valid, metadata
