"""Exercise host creation arguments without cloud calls or host modifications."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "create-host.sh"
MOCK_GCLOUD = r'''
import json, os, pathlib, subprocess, sys
root = pathlib.Path(os.environ["MOCK_ROOT"])
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\n")
if args[:3] == ["compute", "instances", "list"]:
    print(os.environ.get("MOCK_EXISTING_ZONE", ""))
elif args[:3] == ["compute", "instances", "create"]:
    sys.exit(int(args[args.index("--zone") + 1] == os.environ.get("MOCK_FAIL_ZONE")))
elif args[:2] == ["compute", "ssh"]:
    command = args[args.index("--command") + 1]
    command = command.replace("/var/lib/avplumber-host/", str(root) + "/")
    command = command.replace("/proc/sys/kernel/random/boot_id", str(root / "boot_id"))
    result = subprocess.run(["bash", "-c", command])
    # Simulate the current boot finishing setup after the first readiness probe.
    (root / "ready-boot-id").write_text("current-boot\n")
    sys.exit(result.returncode)
else:
    raise AssertionError(args)
'''


class CreateHostTests(unittest.TestCase):
    def run_creator(self, *, stale_ready=False, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, contents in (
                ("gcloud", f"#!{sys.executable}\n{MOCK_GCLOUD}"),
                ("sleep", "#!/bin/sh\nexit 0\n"),
            ):
                path = root / name
                path.write_text(contents)
                path.chmod(0o755)
            (root / "ready").write_text("driver=615.71.09\n")
            (root / "boot_id").write_text("current-boot\n")
            (root / "ready-boot-id").write_text("old-boot\n" if stale_ready else "current-boot\n")
            env = os.environ.copy()
            for key in ("HOST_SETUP_MODE", "IMAGE_NAME", "IMAGE_FAMILY", "IMAGE_PROJECT",
                        "MACHINE_TYPE", "GPU", "DISK_GB", "NVIDIA_DRIVER_VERSION",
                        "DISABLE_MITIGATIONS", "DISABLE_AUTO_UPGRADES"):
                env.pop(key, None)
            env.update(PROJECT="test-project", ZONES="test-zone-a test-zone-b",
                       MOCK_ROOT=directory, PATH=f"{directory}:{env['PATH']}", **overrides)
            result = subprocess.run(["bash", str(SCRIPT), "test-mixer"], env=env,
                                    capture_output=True, text=True, timeout=10)
            log = root / "calls.jsonl"
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def creation(self, **kwargs):
        result, calls = self.run_creator(**kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return next(args for args in calls if args[:3] == ["compute", "instances", "create"])

    def test_prepared_image_defaults(self):
        args = self.creation()
        self.assertEqual(args[args.index("--image-family") + 1], "avplumber-mixer-host")
        self.assertEqual(args[args.index("--image-project") + 1], "test-project")
        self.assertNotIn("--accelerator", args)

    def test_fresh_ubuntu_keeps_security_defaults(self):
        args = self.creation(HOST_SETUP_MODE="install")
        self.assertEqual(args[args.index("--image-family") + 1], "ubuntu-2604-lts-amd64")
        self.assertEqual(args[args.index("--image-project") + 1], "ubuntu-os-cloud")
        self.assertEqual(args[args.index("--boot-disk-size") + 1], "256GB")
        metadata = args[args.index("--metadata") + 1]
        for value in ("host-mode=install", "driver-version=615.71.09",
                      "disable-mitigations=0", "disable-auto-upgrades=0"):
            self.assertIn("avplumber-" + value, metadata)

    def test_image_pin_and_explicit_performance_settings(self):
        args = self.creation(HOST_SETUP_MODE="install", IMAGE_NAME="pinned-image",
                             IMAGE_PROJECT="image-project", DISABLE_MITIGATIONS="1",
                             DISABLE_AUTO_UPGRADES="1")
        self.assertNotIn("--image-family", args)
        self.assertEqual(args[args.index("--image") + 1], "pinned-image")
        metadata = args[args.index("--metadata") + 1]
        self.assertIn("avplumber-disable-mitigations=1", metadata)
        self.assertIn("avplumber-disable-auto-upgrades=1", metadata)

    def test_stale_readiness_waits_for_current_boot(self):
        result, calls = self.run_creator(stale_ready=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(args[:2] == ["compute", "ssh"] for args in calls), 2)

    def test_zone_failure_falls_back(self):
        result, calls = self.run_creator(MOCK_FAIL_ZONE="test-zone-a")
        self.assertEqual(result.returncode, 0, result.stderr)
        zones = [args[args.index("--zone") + 1] for args in calls
                 if args[:3] == ["compute", "instances", "create"]]
        self.assertEqual(zones, ["test-zone-a", "test-zone-b"])

    def test_existing_instance_is_not_duplicated(self):
        result, calls = self.run_creator(MOCK_EXISTING_ZONE="test-zone-a")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(calls), 1)

    def test_invalid_settings_fail_before_cloud_calls(self):
        for overrides in ({"HOST_SETUP_MODE": "unexpected"}, {"DISABLE_MITIGATIONS": "yes"},
                          {"NVIDIA_DRIVER_VERSION": "580.178.04"},
                          {"HOST_SETUP_MODE": "install", "NVIDIA_DRIVER_VERSION": ""}):
            with self.subTest(overrides=overrides):
                result, calls = self.run_creator(**overrides)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
