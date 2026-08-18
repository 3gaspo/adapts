"""Check exact full and cardinality-matched retrieval scopes."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if "einops" not in sys.modules:
    einops = types.ModuleType("einops")
    einops.rearrange = lambda *args, **kwargs: None
    sys.modules["einops"] = einops

from data.neighbors import (  # noqa: E402
    search_neighbors_other_users,
    search_neighbors_other_users_matched,
    search_neighbors_same_user,
)


def _pairwise(values: np.ndarray, store: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.sqrt(((values[:, None] - store[None]) ** 2).sum(axis=-1))
    if metric == "pearson":
        values = values - values.mean(axis=1, keepdims=True)
        store = store - store.mean(axis=1, keepdims=True)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    store = store / np.maximum(np.linalg.norm(store, axis=1, keepdims=True), 1e-8)
    return 1.0 - values @ store.T


def main() -> None:
    queries = np.asarray(
        [[1.0, 0.0, 0.8], [0.0, 1.0, 0.8], [0.8, 0.0, 1.0]],
        dtype=np.float32,
    )
    store = np.asarray(
        [
            [1.0, 0.0, 0.8],
            [0.9, 0.1, 0.7],
            [1.1, -0.1, 0.9],
            [0.0, 1.0, 0.8],
            [0.1, 0.9, 0.7],
            [-0.1, 1.1, 0.9],
            [0.8, 0.0, 1.0],
            [0.7, 0.1, 0.9],
            [0.9, -0.1, 1.1],
        ],
        dtype=np.float32,
    )
    n_users = 3
    store_dates = 3
    for metric in ("euclidean", "cosine", "pearson"):
        same_distances, same_indices = search_neighbors_same_user(
            queries,
            store,
            n_users=n_users,
            store_dates=store_dates,
            k=2,
            metric=metric,
            chunk_size=2,
        )
        other_distances, other_indices = search_neighbors_other_users(
            queries,
            store,
            n_users=n_users,
            store_dates=store_dates,
            k=2,
            metric=metric,
            chunk_size=2,
        )
        matched_distances, matched_indices = search_neighbors_other_users_matched(
            queries,
            store,
            n_users=n_users,
            store_dates=store_dates,
            k=2,
            metric=metric,
            chunk_size=2,
        )
        assert np.isfinite(same_distances).all()
        assert np.isfinite(other_distances).all()
        assert np.isfinite(matched_distances).all()
        expected = _pairwise(queries, store, metric)
        for user_idx in range(n_users):
            own = np.arange(user_idx * store_dates, (user_idx + 1) * store_dates)
            other = np.setdiff1d(np.arange(len(store)), own)
            date_indices = np.arange(store_dates)
            matched_users = (user_idx + 1 + date_indices % (n_users - 1)) % n_users
            matched = matched_users * store_dates + date_indices
            assert all(index in own for index in same_indices[user_idx])
            assert all(index not in own for index in other_indices[user_idx])
            assert all(index in matched for index in matched_indices[user_idx])
            assert all(index not in own for index in matched_indices[user_idx])
            assert len(matched) == len(own) == store_dates
            np.testing.assert_allclose(
                same_distances[user_idx],
                expected[user_idx, same_indices[user_idx]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                other_distances[user_idx],
                expected[user_idx, other_indices[user_idx]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                same_distances[user_idx],
                np.sort(expected[user_idx, own])[:2],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                other_distances[user_idx],
                np.sort(expected[user_idx, other])[:2],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                matched_distances[user_idx],
                np.sort(expected[user_idx, matched])[:2],
                atol=1e-6,
            )

    try:
        search_neighbors_other_users(
            queries[:1],
            store[:store_dates],
            n_users=1,
            store_dates=store_dates,
            k=1,
        )
    except ValueError as error:
        assert "eligible windows" in str(error)
    else:
        raise AssertionError("single-user other-users retrieval must fail")

    try:
        search_neighbors_other_users_matched(
            queries[:1],
            store[:store_dates],
            n_users=1,
            store_dates=store_dates,
            k=1,
        )
    except ValueError as error:
        assert "eligible windows" in str(error)
    else:
        raise AssertionError("single-user matched retrieval must fail")

    print("retrieval scope checks passed")


if __name__ == "__main__":
    main()
