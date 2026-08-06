"""Smoke-check the combined K-ablation candidate plot."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.visu.k_ablation_plot import build_k_ablation_plot


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for family, space, method, offset in (
            ("baselines", "instance", "cov_y_ridge_shared", 0.0),
            ("gates", "raw", "bayes_cov_shared", -1.0),
        ):
            for neighbors, improvement in (
                (1, 1.0),
                (3, 2.0),
                (5, 2.5),
                (10, 3.0),
                (15, 2.8),
                (20, 2.6),
                (100, 1.5),
            ):
                rows.append(
                    {
                        "winner_name": (
                            f"{family}/{space}_euclidean_{neighbors}_online/{method}"
                        ),
                        "family": family,
                        "retrieval": f"{space}_euclidean_{neighbors}_online",
                        "method": method,
                        "average_nmse": 0.9,
                        "average_improvement_pct": improvement + offset,
                    }
                )
        ranking = root / "pipeline_ranking.csv"
        pd.DataFrame(rows).to_csv(ranking, index=False)

        outputs = build_k_ablation_plot(
            ranking,
            root / "plots",
            neighbors=[1, 3, 5, 10, 15, 20, 100],
        )
        assert {path.suffix for path in outputs} == {".csv", ".png", ".pdf"}
        curve = pd.read_csv(root / "plots" / "k_ablation_average_nmse_improvement.csv")
        assert len(curve) == 14
        best = curve.loc[curve["global_best"]]
        assert len(best) == 1
        assert best.iloc[0]["method"] == "cov_y_ridge_shared"
        assert int(best.iloc[0]["neighbors"]) == 10

        try:
            build_k_ablation_plot(
                ranking,
                root / "incomplete",
                neighbors=[1, 3, 5, 10, 15, 20, 50, 100],
            )
        except ValueError as error:
            assert "incomplete K-ablation curves" in str(error)
        else:
            raise AssertionError("incomplete K-ablation grid was accepted")

    print("K-ablation plot checks passed")


if __name__ == "__main__":
    main()
