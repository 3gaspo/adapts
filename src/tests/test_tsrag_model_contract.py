"""Static contracts for source-faithful TS-RAG inference."""

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def parsed(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


class TSRAGModelContractTest(unittest.TestCase):
    def test_model_retains_released_moe_head(self):
        model, tree = parsed(ROOT / "src/models/ts_rag.py")
        classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }

        self.assertIn("ChronosBoltModelForForecastingWithRetrieval", classes)
        self.assertIn("73ac807789d2e61b8a3dfc8514e3fc947fe185cc", model)
        for module in ("encode_mlp", "mha", "ffn", "gate_layer"):
            self.assertIn(f"self.{module}", model)
        self.assertIn("F.softmax(scores, dim=1)", model)
        self.assertGreaterEqual(model.count("scale == 1"), 2)
        for crossrag_module in (
            "encode_mlp_x",
            "encode_mlp_y",
            "cross_mha",
            "self_mha",
        ):
            self.assertNotIn(crossrag_module, model)
        self.assertNotIn("BaseChronosPipeline", model)

    def test_evaluator_strict_loads_public_checkpoint(self):
        evaluator, _ = parsed(ROOT / "src/adaptors/ts_rag/evaluate.py")
        runner = (ROOT / "src/slurm/run_tsrag.sh").read_text(encoding="utf-8")

        self.assertIn("from src.models.ts_rag import", evaluator)
        self.assertIn("strict=True", evaluator)
        self.assertIn("--chronos-bolt-weights", evaluator)
        self.assertIn("--ts-rag-weights", evaluator)
        self.assertIn("--neighbors", evaluator)
        self.assertIn("find_weight_path chronos-bolt-base", runner)
        self.assertIn("find_weight_path ts-rag", runner)

    def test_slurm_front_uses_tsrag_retrieval_defaults(self):
        front = (ROOT / "tsrag.slurm").read_text(encoding="utf-8")
        runner = (ROOT / "src/slurm/run_tsrag.sh").read_text(encoding="utf-8")
        orchestrator = (ROOT / "src/slurm/run_tsrag_experiment.sh").read_text(encoding="utf-8")
        profiles = (ROOT / "src/slurm/profiles.sh").read_text(encoding="utf-8")

        self.assertIn('STAGES="${STAGES:-evaluate,tables}"', front)
        self.assertIn("retrieval_protocol=tsrag_default", runner)
        extraction = (ROOT / "src/slurm/extract_adaptation.sh").read_text(encoding="utf-8")
        self.assertIn("find_weight_path chronos-t5-base", extraction)
        self.assertIn("DISTANCE_SPACES_CSV=tsrag", orchestrator)
        self.assertIn("DISTANCE_METRICS_CSV=euclidean", orchestrator)
        self.assertIn("RETRIEVAL_SCOPE=same_user", orchestrator)
        self.assertIn("NEIGHBORS_CSV=10", orchestrator)
        self.assertIn('DEFAULT_DISTANCE_SPACES_CSV="tsrag"', profiles)

    def test_retriever_uses_chronos_t5_eos_embedding(self):
        retriever, _ = parsed(ROOT / "src/models/tsrag_retriever.py")
        extraction, _ = parsed(ROOT / "src/experiments/extraction.py")
        neighbors, _ = parsed(ROOT / "src/data/neighbors.py")

        self.assertIn("BaseChronosPipeline.from_pretrained", retriever)
        self.assertIn("embeddings[:, -1, :]", retriever)
        self.assertIn('args.retrieval_scope == "same_user"', extraction)
        self.assertIn("def search_neighbors_same_user", neighbors)

    def test_retriever_runtime_options_do_not_change_extraction_identity(self):
        extraction, _ = parsed(ROOT / "src/experiments/extraction.py")
        signature_body = extraction.split("def extraction_signature", 1)[1].split(
            "def _empty_neighbor_tensors", 1
        )[0]

        self.assertIn('"retrieval_scope"', signature_body)
        self.assertNotIn('"retrieval_model_kwargs"', signature_body)
        self.assertNotIn('"representation_batch_size"', signature_body)

    def test_comparison_table_keeps_tsrag_bolt_only(self):
        path = ROOT / "src/visu/tsrag_comparison_table.py"
        spec = importlib.util.spec_from_file_location("tsrag_comparison_table", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controls = root / "controls"
            tsrag = root / "tsrag"
            for backbone in ("chronos2", "chronos-bolt"):
                for dataset in module.DATASETS:
                    run = controls / backbone / dataset / "run_0"
                    run.mkdir(parents=True)
                    (run / "manifest.json").write_text(
                        json.dumps(
                            {
                                "manifest_id": f"control-{backbone}-{dataset}",
                                "status": "completed",
                                "launch": {"launch_id": f"control-{dataset}"},
                                "identity": {"dataset": dataset, "backbone": backbone},
                            }
                        ),
                        encoding="utf-8",
                    )
                    metrics = [
                        {
                            "split": "eval",
                            "baseline": method,
                            "mse": 0.2,
                            "mae": 0.1,
                            "nmse": 0.3,
                            "positive_window_pct": 50.0,
                        }
                        for method in ("vanilla", "our_method")
                    ]
                    (run / "baseline_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            for dataset in module.DATASETS:
                run = tsrag / dataset / "run_0"
                run.mkdir(parents=True)
                (run / "manifest.json").write_text(
                    json.dumps(
                        {
                            "manifest_id": f"tsrag-{dataset}",
                            "status": "completed",
                            "launch": {"launch_id": f"tsrag-{dataset}"},
                            "identity": {"dataset": dataset, "backbone": "chronos-bolt"},
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "tsrag_metrics.json").write_text(
                    json.dumps(
                        [
                            {
                                "mse": 0.19,
                                "mae": 0.09,
                                "nmse": 0.29,
                                "positive_window_pct": 55.0,
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
            paths = module.build(controls, tsrag, root / "report")
            text = paths["csv"].read_text(encoding="utf-8")
            report = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertIn("chronos-bolt,tsrag", text)
            self.assertNotIn("chronos2,tsrag", text)
            self.assertEqual(len(report["obtained_manifests"]), 12)

    def test_comparison_table_reuses_completed_cross_launch_results(self):
        table = (ROOT / "src/visu/tsrag_comparison_table.py").read_text(encoding="utf-8")
        self.assertNotIn("EXPERIMENT_LAUNCH_ID", table)
        self.assertIn('manifest.get("status") == "completed"', table)


if __name__ == "__main__":
    unittest.main()
