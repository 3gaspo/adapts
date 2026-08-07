"""Contract tests for schema-v1 run allocation and table selection."""

import json
import tempfile
import unittest
from pathlib import Path

from experiment_runs import (
    ManifestError,
    allocate_run,
    load_manifest,
    mark_status,
    select_identity_runs,
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


if __name__ == "__main__":
    unittest.main()
