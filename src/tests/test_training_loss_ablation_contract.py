"""Contracts for the MSE-versus-nMSE fitting-objective ablation."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load_module():
    path = ROOT / "src/visu/training_loss_ablation_table.py"
    spec = importlib.util.spec_from_file_location("training_loss_ablation_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainingLossAblationContractTest(unittest.TestCase):
    def test_front_runs_both_objectives_on_selected_trainable_methods(self):
        front = (ROOT / "training_loss_ablation.slurm").read_text(encoding="utf-8")
        orchestrator = (ROOT / "src/slurm/run_profile_experiment.sh").read_text(encoding="utf-8")
        evaluator = (ROOT / "src/adaptors/baselines/evaluate.py").read_text(encoding="utf-8")
        self.assertIn("SWEEP_CANDIDATES.txt", front)
        self.assertIn("for FIT_LOSS in mse nmse", orchestrator)
        self.assertIn('choices=("mse", "nmse")', evaluator)
        self.assertIn("_fit_loss_scale", evaluator)

    def test_table_requires_and_reports_both_objectives(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            identity_root = results / "Electricity" / "method"
            for index, fit_loss in enumerate(module.FIT_LOSSES):
                run = identity_root / f"run_{index}"
                run.mkdir(parents=True)
                (run / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "manifest_id": f"loss-{fit_loss}",
                            "status": "completed",
                            "launch": {"launch_id": fit_loss},
                            "config": {"pipeline": {"fit_loss": fit_loss}},
                            "signatures": {"pipeline": f"pipeline-{fit_loss}"},
                            "purposes": ["publication"],
                            "identity": {
                                "dataset": "Electricity",
                                "lookback": 168,
                                "horizon": 24,
                                "backbone": "chronos2",
                                "model_config": {
                                    "formula": "full_ridge_shared",
                                    "space": "raw",
                                    "metric": "euclidean",
                                    "k": 3,
                                    "mode": "online",
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                metrics = [
                    {
                        "split": "eval",
                        "baseline": "vanilla",
                        "mse": 2.0,
                        "nmse": 1.0,
                        "mae": 1.0,
                        "nmae": 0.5,
                    },
                    {
                        "split": "eval",
                        "baseline": "full_ridge_shared",
                        "mse": 1.0 - 0.1 * index,
                        "nmse": 0.8 - 0.1 * index,
                        "mae": 0.7,
                        "nmae": 0.4,
                    },
                ]
                (run / "baseline_metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            paths = module.build(
                results,
                root / "report",
                pipelines=("baselines/raw_euclidean_3_online/full_ridge_shared",),
                datasets=("Electricity",),
                settings=("168:24",),
                model="chronos2",
                purposes=("publication",),
            )
            text = paths["csv"].read_text(encoding="utf-8")
            self.assertIn(",mse,Electricity,168:24", text)
            self.assertIn(",nmse,Electricity,168:24", text)
            report = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(report["obtained"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
