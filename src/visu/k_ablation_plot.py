"""Plot average held-out improvement against K for every K-ablation candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(
    r"^(?P<space>[^_]+)_(?P<metric>[^_]+)_(?P<neighbors>\d+)_"
    r"(?P<mode>online|fixed)$"
)


def _split_names(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _candidate_label(row: pd.Series) -> str:
    return (
        f"{row['method']} [{row['family']}; "
        f"{row['space']}/{row['distance_metric']}/{row['retrieval_mode']}]"
    )


def build_k_ablation_plot(
    ranking_csv: str | Path,
    output_dir: str | Path,
    *,
    neighbors: Sequence[int],
    metric: str = "nmse",
) -> list[Path]:
    """Build the combined candidate curve and its tidy numeric payload."""
    ranking_path = Path(ranking_csv).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    frame = pd.read_csv(ranking_path)
    required = {
        "family",
        "retrieval",
        "method",
        f"average_{metric}",
        "average_improvement_pct",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{ranking_path} is missing columns: {missing_columns}")

    parsed = frame["retrieval"].str.extract(RUN_PATTERN)
    if parsed.isna().any(axis=None):
        invalid = frame.loc[parsed.isna().any(axis=1), "retrieval"].tolist()
        raise ValueError(f"cannot parse K-ablation retrieval names: {invalid}")
    parsed = parsed.rename(
        columns={"metric": "distance_metric", "mode": "retrieval_mode"}
    )
    frame = pd.concat([frame, parsed], axis=1)
    frame["neighbors"] = frame["neighbors"].astype(int)
    frame["candidate"] = frame.apply(_candidate_label, axis=1)

    expected_neighbors = tuple(sorted({int(value) for value in neighbors}))
    expected_set = set(expected_neighbors)
    incomplete = {
        candidate: sorted(expected_set - set(group["neighbors"]))
        for candidate, group in frame.groupby("candidate")
        if set(group["neighbors"]) != expected_set
    }
    if incomplete:
        raise ValueError(f"incomplete K-ablation curves: {incomplete}")

    frame = frame.sort_values(["candidate", "neighbors"]).reset_index(drop=True)
    best_index = frame["average_improvement_pct"].astype(float).idxmax()
    frame["global_best"] = False
    frame.loc[best_index, "global_best"] = True

    destination.mkdir(parents=True, exist_ok=True)
    stem = f"k_ablation_average_{metric}_improvement"
    csv_path = destination / f"{stem}.csv"
    output_columns = [
        "candidate",
        "family",
        "space",
        "distance_metric",
        "retrieval_mode",
        "method",
        "neighbors",
        f"average_{metric}",
        "average_improvement_pct",
        "global_best",
    ]
    frame.to_csv(csv_path, columns=output_columns, index=False)

    fig, ax = plt.subplots(figsize=(14, 7.5))
    colors = plt.get_cmap("tab10").colors
    for index, (candidate, values) in enumerate(frame.groupby("candidate", sort=True)):
        ax.plot(
            values["neighbors"],
            values["average_improvement_pct"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=colors[index % len(colors)],
            label=candidate,
        )

    best = frame.loc[best_index]
    ax.scatter(
        [best["neighbors"]],
        [best["average_improvement_pct"]],
        marker="*",
        s=190,
        color="black",
        zorder=5,
        label=(
            f"Global best: {best['method']}, K={int(best['neighbors'])} "
            f"({float(best['average_improvement_pct']):+.2f}%)"
        ),
    )
    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(expected_neighbors, [str(value) for value in expected_neighbors])
    ax.set_xlabel("Number of retrieved neighbours K")
    ax.set_ylabel(f"Average {metric.upper()} improvement over vanilla (%)")
    ax.set_title(f"K ablation: average {metric.upper()} improvement")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    outputs = [csv_path]
    for suffix in ("png", "pdf"):
        path = destination / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbors", required=True)
    parser.add_argument("--metric", default="nmse")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    outputs = build_k_ablation_plot(
        args.ranking_csv,
        args.output_dir,
        neighbors=[int(value) for value in _split_names(args.neighbors)],
        metric=args.metric,
    )
    for output in outputs:
        print(f"K-ablation plot artifact written to {output}")
    return outputs


if __name__ == "__main__":
    main()
