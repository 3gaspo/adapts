"""Static contract checks for exact-path Slurm publishing."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PublisherContractTest(unittest.TestCase):
    def test_publisher_is_exact_afterok_and_proxy_aware(self):
        publisher = (ROOT / "src/slurm/publish_results.sh").read_text(encoding="utf-8")
        front = (ROOT / "publish.slurm").read_text(encoding="utf-8")
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        shells = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src/slurm").glob("*.sh")
        )

        self.assertIn('"--dependency=afterok:$producer_job_id"', publisher)
        self.assertIn("logs/%s_%s.out", publisher)
        self.assertIn("logs/%s_%s.err", publisher)
        self.assertIn("launch_id", publisher)
        self.assertIn("git add -v -f --", publisher)
        self.assertIn("git commit --only", publisher)
        self.assertIn("git push origin main", publisher)
        self.assertNotIn("git pull", publisher)
        self.assertIn("**/*.pt", publisher)
        self.assertIn("**/*.npy", publisher)
        self.assertIn("**/*.cbm", publisher)
        self.assertIn('. "$proxy_script" --credentials-file "$credentials_file"', publisher)
        self.assertIn("$HOME/codes/proxy.sh", publisher)
        self.assertIn("$HOME/codes/.secrets/proxy.credentials", publisher)
        self.assertIn("publish_results_main", front)
        self.assertIn("submit_publish_job", shells)
        self.assertIn(".secrets/", ignore)
        self.assertIn("experiment_runs complete-launch", shells)
        self.assertIn("mark_manifest_ready", shells)
        self.assertNotIn("mark_manifest_completed", shells)


if __name__ == "__main__":
    unittest.main()
