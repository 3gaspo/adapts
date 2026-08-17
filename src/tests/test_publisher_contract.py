"""Static contract checks for log-scoped or full-tree manual publishing."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PublisherContractTest(unittest.TestCase):
    def test_publisher_is_manual_scoped_and_proxy_aware(self):
        publisher = (ROOT / "publish_job.sh").read_text(encoding="utf-8")
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        common = (ROOT / "src/slurm/common.sh").read_text(encoding="utf-8")

        self.assertFalse((ROOT / "publish.slurm").exists())
        self.assertFalse((ROOT / "src/slurm/publish_results.sh").exists())
        self.assertFalse((ROOT / "src/scripts/publish_job.sh").exists())
        self.assertIn('logs/*_"$job_id".out', publisher)
        self.assertIn('logs/*_"$job_id".err', publisher)
        self.assertNotIn("launch_id", publisher)
        self.assertNotIn('find "$project_root/outputs"', publisher)
        self.assertIn("paths=(logs outputs)", publisher)
        self.assertIn('if [ -n "$job_id" ]', publisher)
        self.assertIn("git add -v -f --", publisher)
        self.assertIn("git commit --only", publisher)
        self.assertIn("git push origin main", publisher)
        self.assertIn("git pull --ff-only origin main", publisher)
        self.assertLess(publisher.index('. "$proxy_script"'), publisher.index("git pull --ff-only origin main"))
        self.assertLess(publisher.index("git pull --ff-only origin main"), publisher.index("git add -v -f --"))
        self.assertLess(publisher.index("git add -v -f --"), publisher.index("git commit --only"))
        self.assertIn("**/*.pt", publisher)
        self.assertIn("**/*.npy", publisher)
        self.assertIn("**/*.cbm", publisher)
        self.assertIn('. "$proxy_script"', publisher)
        self.assertIn("$HOME/codes/proxy.sh", publisher)
        self.assertNotIn("PROXY_CREDENTIALS_FILE", publisher)
        self.assertNotIn("unset GIT_ASKPASS", publisher)
        self.assertNotIn("submit_publish_job", common)
        self.assertNotIn("ADAPTATION_PUBLISH_SUBMITTED", common)
        self.assertIn(".secrets/", ignore)
        self.assertIn("experiment_runs complete-launch", common)
        self.assertIn("experiment_runs complete --run-dir", common)


if __name__ == "__main__":
    unittest.main()
