"""Exercise launcher arguments with a fake build/runtime; never starts DDS."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


DEPLOY = Path(__file__).resolve().parents[4] / "deploy.sh"


class DeployHandTypeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(DEPLOY, self.root / "deploy.sh")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/setup_env.sh").write_text(":\n")
        (self.root / "scripts/install_deps.sh").write_text("exit 99\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ("cmake", "clang", "git", "ip"):
            self.stub(tool, "exit 0\n")
        self.stub("just", 'printf "%s\\n" "$@" >> "$DEX1_TEST_ARGS"\n')
        self.log = self.root / "args.txt"
        self.env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin",
                        TensorRT_ROOT=str(self.root), DEX1_TEST_ARGS=str(self.log))

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def launch(self, *args):
        return subprocess.run(
            ["bash", str(self.root / "deploy.sh"), *args],
            input="y\n", text=True, capture_output=True, env=self.env, timeout=10,
        )

    def test_dex1_preserves_v11_options(self):
        result = self.launch(
            "--hand-type", "dex1", "--input-type", "zmq_manager",
            "--cp", "policy/sonic_v1_1/model",
            "--obs-config", "policy/sonic_v1_1/observation_config.yaml",
            "--motor-kp-scale", "4,10=1.5", "--motor-kd-scale", "4,10=1.5", "sim",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.log.read_text().splitlines()
        for flag, value in (
            ("--hand-type", "dex1"), ("--input-type", "zmq_manager"),
            ("--motor-kp-scale", "4,10=1.5"), ("--motor-kd-scale", "4,10=1.5"),
            ("--obs-config", "policy/sonic_v1_1/observation_config.yaml"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("policy/sonic_v1_1/model_decoder.onnx", args)
        self.assertIn("policy/sonic_v1_1/model_encoder.onnx", args)
        self.assertIn("--disable-crc-check", args)

    def test_default_remains_dex3(self):
        result = self.launch("sim")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.log.read_text().splitlines()
        self.assertEqual(args[args.index("--hand-type") + 1], "dex3")

    def test_none(self):
        result = self.launch("--hand-type", "none", "sim")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.log.read_text().splitlines()
        self.assertEqual(args[args.index("--hand-type") + 1], "none")

    def test_invalid_or_missing_type_fails_before_build(self):
        for args in (("--hand-type",), ("--hand-type", "bad")):
            with self.subTest(args=args):
                result = self.launch(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--hand-type requires", result.stderr)
                self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
