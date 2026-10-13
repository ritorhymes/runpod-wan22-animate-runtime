"""Free, CPU-only official Wan Animate runtime contract regression tests."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / "scripts" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("wan22_runtime", FILE)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ModelManifestContractTests(unittest.TestCase):
    def test_manifest_is_pinned_and_code_only(self):
        m = mod.manifest()
        self.assertEqual(m["repo_id"], "Wan-AI/Wan2.2-Animate-14B")
        self.assertEqual(m["revision"], "cb93a225fbaf1ca100f54e79da8f994995b689b3")
        self.assertEqual(len(m["generation_files"]), 19)
        self.assertEqual(len(m["preprocess_files"]), 395)
        self.assertEqual(
            sum(x["size"] for x in m["generation_files"]),
            m["generation_total_bytes"],
        )
        self.assertEqual(
            sum(x["size"] for x in m["preprocess_files"]),
            m["preprocess_total_bytes"],
        )
        for item in mod.all_files(True):
            self.assertFalse(Path(item["path"]).is_absolute())
            self.assertNotIn("..", Path(item["path"]).parts)
            self.assertGreaterEqual(item["size"], 1)
            if item.get("sha256"):
                self.assertEqual(len(item["sha256"]), 64)

    def test_plan_is_read_only(self):
        with patch.object(mod, "download_models", side_effect=AssertionError("Paid/download")):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                mod.print_plan(False)
        self.assertIn("MODEL_PLAN_OFFLINE", stdout.getvalue())
        self.assertIn("54,100,856,515", stdout.getvalue())
        self.assertNotIn("process_checkpoint/det/yolov10m.onnx", stdout.getvalue())

    def test_size_and_checksum_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "tiny.bin").write_bytes(b"official bytes")
            from hashlib import sha256
            sha = sha256(b"official bytes").hexdigest()
            specs = [{"path": "tiny.bin", "size": len(b"official bytes"),
                      "sha256": sha}]
            with patch.object(mod, "all_files", return_value=specs):
                passed = mod.verify_files(p, verify_sha=True)
                self.assertTrue(passed["complete"])
                (p / "tiny.bin").write_bytes(b"modified bytes")
                fail = mod.verify_files(p, verify_sha=True)
                self.assertFalse(fail["complete"])
                self.assertTrue(any("WRONG SHA256" in e for e in fail["errors"]))

    def test_relative_paths_must_not_escape_model_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(mod, "all_files",
                              return_value=[{"path": "../secret", "size": 99, "sha256": None}]):
                with self.assertRaisesRegex(mod.RuntimeErrorContract, "escapes root"):
                    mod.verify_files(Path(directory))


class ConditionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = Path(self.temp.name)
        for asset in mod.ASSETS:
            (self.p / asset).write_bytes(b"test bytes")
        from PIL import Image
        Image.new("RGB", (512, 700)).save(self.p / "src_ref.png")
        self.info = {
            "mode": "official-wan22-animate-replace",
            "source_fps": "24000/1001",
            "source_frame_count": 40,
            "generation_fps": 30,
            "generation_frames": 77,
            "motion_frames": 51,
            "condition_frame_to_source_local_frame": (
                [min(39, i * 24000 // (30 * 1001)) for i in range(77)]
            ),
        }
        (self.p / "job.json").write_text(json.dumps(self.info))
        self.probe = {
            "codec_name": "h264",
            "codec_type": "video",
            "width": 832,
            "height": 480,
            "avg_frame_rate": "30/1",
            "nb_read_frames": "77",
        }

    def test_correct_preflight_and_wrong_size_rejected(self):
        def fake_ffprobe(path):
            base = self.probe.copy()
            if path.name == "src_face.mp4":
                base.update(width=512, height=512)
            return base
        with patch.object(mod, "ffprobe", side_effect=fake_ffprobe):
            details = mod.preflight_job(self.p)
            self.assertEqual(details["motion_frames"], 51)

        wrong = self.probe.copy()
        wrong["nb_read_frames"] = "76"
        with patch.object(mod, "ffprobe", return_value=wrong):
            with self.assertRaisesRegex(mod.RuntimeErrorContract, "77 frames"):
                mod.preflight_job(self.p)

    def test_audio_mapping_must_be_monotonic_and_original_duration_retained(self):
        self.info["condition_frame_to_source_local_frame"][44] = 0
        (self.p / "job.json").write_text(json.dumps(self.info))
        with self.assertRaisesRegex(mod.RuntimeErrorContract, "monotonic"):
            mod.preflight_job(self.p)

    def test_cpu_safe_infer_dryrun_never_calls_gpu(self):
        with patch.object(mod, "preflight_job"), \
             patch.object(mod, "verify_files", return_value={"complete": True}), \
             patch.object(mod, "smoke", side_effect=AssertionError("smoke called")), \
             patch.object(mod.subprocess, "run", side_effect=AssertionError("subprocess started")):
            with contextlib.redirect_stdout(io.StringIO()):
                argv = mod.run(
                    self.p, Path("/tmp/unused-models"), Path("/tmp/unused-result.mp4"),
                    20, 42, actually_run=False,
                )
        self.assertEqual(argv[argv.index("--task") + 1], "animate-14B")
        self.assertEqual(argv[argv.index("--sample_steps") + 1], "20")
        self.assertIn("--replace_flag", argv)
        self.assertIn("--use_relighting_lora", argv)
        self.assertIn("--convert_model_dtype", argv)
        self.assertIn("--t5_cpu", argv)
        self.assertEqual(argv[argv.index("--frame_num") + 1], "77")


if __name__ == "__main__":
    unittest.main()
