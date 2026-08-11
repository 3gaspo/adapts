"""Static contracts for the local Chronos-Bolt and Cross-RAG sources."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def parsed(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def class_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


class CrossRAGModelContractTest(unittest.TestCase):
    def test_chronos_bolt_is_a_distinct_local_source(self):
        chronos_text, chronos_tree = parsed(ROOT / "src/models/chronos_model.py")
        bolt_text, bolt_tree = parsed(ROOT / "src/models/chronos_bolt.py")

        self.assertNotIn("ChronosBolt", class_names(chronos_tree))
        self.assertIn("Chronos", class_names(chronos_tree))
        self.assertIn("ChronosBoltModelForForecasting", class_names(bolt_tree))
        self.assertIn("ChronosBolt", class_names(bolt_tree))
        self.assertIn("7dc4435706a4454feb79df44ca9f33631f3027bf", bolt_text)
        self.assertNotIn("BaseChronosPipeline", bolt_text)
        self.assertNotIn("class ChronosBolt", chronos_text)

    def test_cross_rag_retains_released_attention_modules(self):
        model_text, model_tree = parsed(ROOT / "src/models/cross_rag.py")
        classes = class_names(model_tree)

        self.assertIn("ChronosBoltModelForForecastingWithRetrieval", classes)
        self.assertIn("b9a5428365b8ada43a986b2501ece12dd3844e95", model_text)
        for module in (
            "encode_mlp_x",
            "encode_mlp_y",
            "cross_mha",
            "self_mha",
            "ffn_cross",
            "ffn_self",
        ):
            self.assertIn(f"self.{module}", model_text)
        self.assertIn('self.augment == "moe"', model_text)
        self.assertNotIn("BaseChronosPipeline", model_text)

    def test_evaluator_uses_only_local_code_and_weight_paths(self):
        evaluator, _ = parsed(ROOT / "src/adaptors/cross_rag/evaluate.py")
        runner = (ROOT / "src/slurm/run_crossrag.sh").read_text(encoding="utf-8")

        self.assertIn("from src.models.cross_rag import", evaluator)
        self.assertNotIn("importlib", evaluator)
        self.assertNotIn("crossrag-root", evaluator)
        self.assertIn("--chronos-bolt-weights", evaluator)
        self.assertIn("--cross-rag-weights", evaluator)
        self.assertIn("find_weight_path chronos-bolt-base", runner)
        self.assertIn("find_weight_path cross-rag", runner)
        self.assertNotIn("CROSSRAG_ROOT", runner)


if __name__ == "__main__":
    unittest.main()
