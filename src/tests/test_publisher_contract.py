"""Static contract checks for exact-path manual publishing."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PublisherContractTest(unittest.TestCase):
    def test_publisher_is_manual_exact_and_proxy_aware(self):
        publisher = (ROOT / "publish_job.sh").read_text(encoding="utf-8")
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        common = (ROOT / "src/slurm/common.sh").read_text(encoding="utf-8")

        self.assertFalse((ROOT / "publish.slurm").exists())
        self.assertFalse((ROOT / "src/slurm/publish_results.sh").exists())
        self.assertFalse((ROOT / "src/scripts/publish_job.sh").exists())
        self.assertIn('logs/*_"$job_id".out', publisher)
        self.assertIn('logs/*_"$job_id".err', publisher)
        self.assertIn("launch_id", publisher)
        self.assertIn("paths=(logs outputs)", publisher)
        self.assertIn('if [ -n "$job_id" ]', publisher)
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
        self.assertIn("unset GIT_ASKPASS", publisher)
        self.assertNotIn("submit_publish_job", common)
        self.assertNotIn("ADAPTATION_PUBLISH_SUBMITTED", common)
        self.assertIn(".secrets/", ignore)
        self.assertIn("experiment_runs complete-launch", common)


if __name__ == "__main__":
    unittest.main()
