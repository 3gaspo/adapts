"""Contract tests for schema-v1 run allocation and table selection."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from experiment_runs import (
    ManifestError,
    allocate_run,
    complete_launch,
    load_manifest,
    mark_ready,
    mark_status,
    prepare_run_output,
    select_identity_runs,
    validate_completed,
    write_report_manifest,
)


class ExperimentRunsTest(unittest.TestCase):
    def _allocate(self, identity: Path, steps: int, **kwargs):
        return allocate_run(
            identity,
            project="contract_test",
            workflow="family",
            dataset="electricity",
            lookback=504,
            horizon=168,
            backbone="patchtst",
            model_config_order=["formula", "space"],
            model_config={"formula": "ridge", "space": "instance"},
            pipeline_config={"steps": steps},
            seeds=[1],
            display_name="ridge",
            launch_id=f"launch_{steps}_{kwargs.get('policy', 'default')}",
            **kwargs,
        )

    @staticmethod
    def _complete(allocation) -> None:
        relative = "seed_1/result.json"
        artifact = allocation.run_dir / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"mse": 1.0}\n', encoding="utf-8")
        mark_status(
            allocation.run_dir,
            "completed",
            seed=1,
            required_artifacts=[relative],
        )
        assert load_manifest(allocation.run_dir)["status"] == "running"
        mark_status(
            allocation.run_dir,
            "completed",
            required_artifacts=[relative],
        )

    def test_collision_and_selection_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = Path(folder) / "electricity/504_168/patchtst/ridge/instance"

            first = self._allocate(identity, 10)
            self.assertEqual((first.run_dir.name, first.action), ("run_0", "new"))
            self._complete(first)

            identical = self._allocate(identity, 10)
            self.assertEqual((identical.run_dir.name, identical.action), ("run_0", "skip"))

            changed = self._allocate(identity, 20)
            self.assertEqual((changed.run_dir.name, changed.action), ("run_1", "new"))
            self._complete(changed)

            choices = select_identity_runs(identity)
            self.assertEqual(len(choices), 2)
            self.assertTrue(all("__steps-" in choice.label for choice in choices))

            filtered = select_identity_runs(
                identity,
                requested_pipeline={"steps": 20},
            )
            self.assertEqual([choice.run_dir.name for choice in filtered], ["run_1"])

            repeat = self._allocate(identity, 20, policy="new")
            self.assertEqual(repeat.run_dir.name, "run_2")
            self._complete(repeat)
            selected = select_identity_runs(
                identity,
                requested_pipeline={"steps": 20},
                repeat_policy="selected",
            )
            self.assertEqual([choice.run_dir.name for choice in selected], ["run_2"])
            distinct = select_identity_runs(
                identity,
                requested_pipeline={"steps": 20},
                repeat_policy="distinct",
            )
            self.assertEqual(
                [choice.run_dir.name for choice in distinct],
                ["run_1", "run_2"],
            )

    def test_obsolete_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run_0"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps({"schema_version": 0, "status": "completed"}),
                encoding="utf-8",
            )
            with self.assertRaises(ManifestError):
                load_manifest(run)

    def test_provenance_does_not_define_computation(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = Path(folder) / "electricity/504_168/patchtst/ridge/instance"
            first = self._allocate(identity, 10, inputs={"dataset": "old/location.csv"})
            self._complete(first)

            reused = self._allocate(identity, 10, inputs={"dataset": "new/location.csv"})
            self.assertEqual((reused.run_dir.name, reused.action), ("run_0", "skip"))

    def test_allocation_reclaims_manifestless_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = Path(folder) / "electricity/504_168/patchtst/ridge/instance"
            orphan = identity / "run_0"
            orphan.mkdir(parents=True)
            stale = orphan / "result.json"
            stale.write_text('{"stale": true}\n', encoding="utf-8")

            allocation = self._allocate(identity, 10)

            self.assertEqual((allocation.run_dir.name, allocation.action), ("run_0", "new"))
            self.assertFalse(stale.exists())
            self.assertEqual(load_manifest(allocation.run_dir)["status"], "not_run")

    def test_prepare_preserves_manifest_history_and_completed_seed_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = Path(folder) / "electricity/504_168/patchtst/ridge/instance"
            allocation = allocate_run(
                identity,
                project="contract_test",
                workflow="family",
                dataset="electricity",
                lookback=504,
                horizon=168,
                backbone="patchtst",
                model_config_order=["formula", "space"],
                model_config={"formula": "ridge", "space": "instance"},
                pipeline_config={"steps": 10},
                seeds=[1, 2],
                launch_id="launch_prepare",
            )
            completed = allocation.run_dir / "seed_1/result.json"
            completed.parent.mkdir()
            completed.write_text('{"mse": 1.0}\n', encoding="utf-8")
            stale = allocation.run_dir / "seed_2/partial.json"
            stale.parent.mkdir()
            stale.write_text('{"partial": true}\n', encoding="utf-8")
            history = allocation.run_dir / "manifest_history/prior.json"
            history.parent.mkdir()
            history.write_text("{}\n", encoding="utf-8")
            mark_status(
                allocation.run_dir,
                "completed",
                seed=1,
                required_artifacts=["seed_1/result.json"],
            )

            prepare_run_output(allocation.run_dir)

            self.assertTrue((allocation.run_dir / "manifest.json").is_file())
            self.assertTrue(history.is_file())
            self.assertTrue(completed.is_file())
            self.assertFalse(stale.exists())

    def test_ready_run_completes_only_after_slurm_workflow_success(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = Path(folder) / "electricity/504_168/patchtst/ridge/instance"
            allocation = self._allocate(identity, 10)
            mark_status(allocation.run_dir, "running")
            artifact = allocation.run_dir / "result.json"
            artifact.write_text('{"mse": 1.0}\n', encoding="utf-8")

            mark_ready(allocation.run_dir, required_artifacts=["result.json"])
            ready = load_manifest(allocation.run_dir)
            self.assertEqual(ready["status"], "running")
            self.assertEqual(ready["seed_status"]["1"]["status"], "ready")
            self.assertEqual(complete_launch(folder, "launch_10_default"), [allocation.run_dir])
            self.assertEqual(load_manifest(allocation.run_dir)["status"], "completed")

            artifact.unlink()
            self.assertEqual(validate_completed(allocation.run_dir)["status"], "completed")

    def test_report_manifest_records_requested_and_obtained(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = Path(folder) / "electricity/504_168/patchtst/ridge/instance"
            allocation = self._allocate(identity, 10)
            self._complete(allocation)
            selected = select_identity_runs(identity, requested_pipeline={"steps": 10})
            destination = Path(folder) / "report_manifest.json"
            previous_launch = os.environ.get("EXPERIMENT_LAUNCH_ID")
            os.environ["EXPERIMENT_LAUNCH_ID"] = "report_launch"
            try:
                write_report_manifest(
                    destination,
                    inputs=selected,
                    config_policy="distinct",
                    repeat_policy="selected",
                    filters={"pipeline": {"steps": 10}},
                )
            finally:
                if previous_launch is None:
                    os.environ.pop("EXPERIMENT_LAUNCH_ID", None)
                else:
                    os.environ["EXPERIMENT_LAUNCH_ID"] = previous_launch
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(report["launch_id"], "report_launch")
            self.assertEqual(report["requested"]["filters"]["pipeline"], {"steps": 10})
            self.assertEqual(report["obtained"]["count"], 1)
            self.assertEqual(report["obtained"]["inputs"][0]["pipeline_config"], {"steps": 10})


if __name__ == "__main__":
    unittest.main()
