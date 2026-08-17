"""Contracts for the static-paper and exact-evaluation SOTA benchmark."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load_table_module():
    path = ROOT / "src/visu/sota_benchmark_table.py"
    spec = importlib.util.spec_from_file_location("sota_benchmark_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SOTABenchmarkContractTest(unittest.TestCase):
    def test_published_rows_and_protocol_are_explicit(self):
        config = json.loads((ROOT / "SOTA_BENCHMARK.json").read_text(encoding="utf-8"))
        self.assertEqual(config["metric"], "mse")
        self.assertEqual(config["protocol"]["lags"], 512)
        self.assertEqual(config["protocol"]["horizon"], 64)
        self.assertEqual(config["published_results"]["TS-RAG"]["Average"], 0.197)
        self.assertEqual(config["published_results"]["Cross-RAG"]["Average"], 0.191)
        self.assertEqual(config["datasets"]["ETTh1"]["official_boundaries"], [8640, 11520, 14400])
        self.assertEqual(config["datasets"]["ETTh1"]["project_split_bounds"], [4320, 11520, 14400])
        self.assertEqual(config["datasets"]["ETTm1"]["official_boundaries"], [34560, 46080, 57600])
        self.assertEqual(config["datasets"]["ETTm1"]["project_split_bounds"], [17280, 46080, 57600])

    def test_exact_ett_test_window_counts(self):
        self.assertEqual(14400 - 11520 - 64 + 1, 2817)
        self.assertEqual(57600 - 46080 - 64 + 1, 11457)
        neighbors = (ROOT / "src/data/neighbors.py").read_text(encoding="utf-8")
        self.assertIn("int(period_start) - 1", neighbors)
        self.assertIn("int(period_end) - int(horizon) - 1", neighbors)

    def test_front_uses_static_rows_and_never_runs_crossrag(self):
        self.assertFalse((ROOT / "crossrag.slurm").exists())
        front = (ROOT / "sota_benchmark.slurm").read_text(encoding="utf-8")
        runner = (ROOT / "src/slurm/run_sota_benchmark.sh").read_text(encoding="utf-8")
        table = (ROOT / "src/visu/sota_benchmark_table.py").read_text(encoding="utf-8")
        self.assertIn("run_sota_benchmark.sh", front)
        self.assertIn("#SBATCH --mem=80000", front)
        self.assertIn("SOTA_BENCHMARK.json", runner)
        self.assertIn("EVAL_QUERY_STRIDE=1", runner)
        self.assertNotIn("run_crossrag.sh", runner)
        self.assertIn('config["published_results"]', table)

    def test_table_combines_computed_and_published_rows(self):
        config_path = ROOT / "SOTA_BENCHMARK.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        module = load_table_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "report"
            for values in config["datasets"].values():
                run = results / values["project_name"] / "run_0"
                run.mkdir(parents=True)
                (run / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "manifest_id": f"sota-{values['project_name']}",
                            "status": "completed",
                            "launch": {"launch_id": f"source-{values['project_name']}"},
                            "config": {"pipeline": {"run_seed": 1}},
                            "signatures": {"pipeline": "pipeline-1"},
                            "purposes": ["publication"],
                            "identity": {
                                "dataset": values["project_name"],
                                "lookback": 512,
                                "horizon": 64,
                                "backbone": "chronos-bolt",
                                "model_config": {
                                    "formula": "full_ridge_shared",
                                    "space": "instance",
                                    "metric": "euclidean",
                                    "k": 3,
                                    "mode": "online",
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "baseline_metrics.json").write_text(
                    json.dumps(
                        [
                            {"split": "eval", "baseline": "vanilla", "mse": 0.3},
                            {"split": "eval", "baseline": "our_method", "mse": 0.2},
                        ]
                    ),
                    encoding="utf-8",
                )
            stale = results / "obsolete" / "run_0"
            stale.mkdir(parents=True)
            (stale / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "manifest_id": "obsolete-k10",
                        "status": "completed",
                        "launch": {"launch_id": "obsolete"},
                        "config": {"pipeline": {"run_seed": 1}},
                        "signatures": {"pipeline": "pipeline-1"},
                        "purposes": ["publication"],
                        "identity": {
                            "dataset": "ETTh1",
                            "lookback": 512,
                            "horizon": 64,
                            "backbone": "chronos-bolt",
                            "model_config": {
                                "formula": "full_ridge_shared",
                                "space": "instance",
                                "metric": "euclidean",
                                "k": 10,
                                "mode": "online",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (stale / "baseline_metrics.json").write_text(
                json.dumps(
                    [
                        {"split": "eval", "baseline": "vanilla", "mse": 9.0},
                        {"split": "eval", "baseline": "our_method", "mse": 9.0},
                    ]
                ),
                encoding="utf-8",
            )
            selected_pipeline = {
                "formula": "full_ridge_shared",
                "space": "instance",
                "metric": "euclidean",
                "k": 3,
                "mode": "online",
            }
            paths = module.build_table(
                config_path,
                results,
                output,
                selected_pipeline,
                purposes=("publication",),
            )
            text = paths["csv"].read_text(encoding="utf-8")
            report = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertIn("our_method", text)
            self.assertIn("TS-RAG (published)", text)
            self.assertIn("Cross-RAG (published)", text)
            self.assertEqual(report["obtained"]["count"], 7)
            self.assertEqual(
                report["requested"]["filters"]["selected_pipeline"], selected_pipeline
            )
            self.assertEqual(
                report["requested"]["filters"]["purposes"], ["publication"]
            )

    def test_table_reuses_completed_cross_launch_results(self):
        table = (ROOT / "src/visu/sota_benchmark_table.py").read_text(encoding="utf-8")
        self.assertNotIn("EXPERIMENT_LAUNCH_ID", table)
        self.assertIn("selected_manifest_runs", table)
        self.assertIn("write_report_manifest", table)
        self.assertNotIn('rglob("manifest.json")', table)


if __name__ == "__main__":
    unittest.main()
