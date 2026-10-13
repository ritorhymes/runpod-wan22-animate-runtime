#!/usr/bin/env python3
"""Official Wan2.2-Animate-14B disposable Pod runner. NO Pod-creation capability.

Read-only commands (plan/verify/preflight/smoke) are safe without a GPU.
Models must be downloaded explicitly and the separate private controller must
be used to create/terminate paid Pods after user authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent.parent
MANIFEST = RUNTIME / "model-manifest.json"
UPSTREAM = Path(os.environ.get("WAN22_UPSTREAM", "/opt/wan2.2"))
DEFAULT_MODELS = Path(os.environ.get("WAN22_MODELS", "/workspace/models/Wan2.2-Animate-14B"))
ASSETS = ("src_ref.png", "src_pose.mp4", "src_face.mp4",
          "src_mask.mp4", "src_bg.mp4")
SAMPLE_FPS = Fraction(30, 1)
SAMPLE_FRAMES = 77
SIZE = (832, 480)
FACE_SIZE = (512, 512)


class RuntimeErrorContract(RuntimeError):
    pass


def manifest() -> dict:
    data = json.loads(MANIFEST.read_text())
    assert data["schema_version"] == 1 and len(data["revision"]) == 40
    assert data["repo_id"] == "Wan-AI/Wan2.2-Animate-14B"
    return data


def all_files(include_preprocess: bool = False) -> list[dict]:
    m = manifest()
    return m["generation_files"] + (m["preprocess_files"] if include_preprocess else [])


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_files(root: Path, *, include_preprocess: bool = False,
                 verify_sha: bool = False) -> dict:
    total = 0
    errors = []
    for item in all_files(include_preprocess):
        rel = item["path"]
        dest = (root / rel).resolve()
        if not dest.is_relative_to(root.resolve()):
            raise RuntimeErrorContract(f"Manifest path escapes root: {rel}")
        if not dest.is_file():
            errors.append(f"MISSING {rel}")
            continue
        size = dest.stat().st_size
        if size != item["size"]:
            errors.append(f"WRONG SIZE {rel}: {size} vs {item['size']}")
        elif verify_sha and item.get("sha256") and digest(dest) != item["sha256"]:
            errors.append(f"WRONG SHA256 {rel}")
        else:
            total += size
    return {
        "files": len(all_files(include_preprocess)),
        "verified_size_bytes": total,
        "sha256_checked": verify_sha,
        "errors": errors,
        "complete": not errors
    }


def print_plan(include_preprocess: bool):
    m = manifest()
    selected = all_files(include_preprocess)
    print("MODEL_PLAN_OFFLINE (NO DOWNLOAD, NO POD)")
    print(f"Repo: {m['repo_id']} pinned at {m['revision']}")
    print(f"Files: {len(selected)}; total bytes {sum(x['size'] for x in selected):,}")
    print(f"Generation-only files: {len(m['generation_files'])}; bytes {m['generation_total_bytes']:,}")
    if include_preprocess:
        print(f"Preprocessor files: {len(m['preprocess_files'])}; bytes {m['preprocess_total_bytes']:,}")
    print("Selected model paths:")
    for item in selected:
        if include_preprocess and item["path"].startswith("process_checkpoint/pose2d/") \
                and not item["path"].endswith("end2end.onnx"):
            continue
        print(f"  {item['size']:>14,}  {item['path']}")
    print("No model weights are present in or built into the GHCR image.")


def download_models(root: Path, include_preprocess: bool, workers: int):
    """Explicit only, never automatically called by infer/preflight/smoke."""
    from huggingface_hub import snapshot_download

    m = manifest()
    needed = all_files(include_preprocess)
    required = sum(f["size"] for f in needed)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    # HF local snapshot can temporarily cache blobs, so reserve extra space.
    if free < int(required * 1.30):
        raise RuntimeErrorContract(
            f"Not enough DISPOSABLE container disk: free {free:,}; "
            f"need at least {int(required * 1.30):,} bytes"
        )
    print("DOWNLOAD_START", datetime.now(timezone.utc).isoformat(), flush=True)
    print(f"Official pinned repo {m['repo_id']} rev {m['revision']}", flush=True)
    print(f"Files={len(needed)} expected_size={required:,} "
          f"root={root} (disposable filesystem)", flush=True)
    t0 = time.monotonic()
    path = snapshot_download(
        repo_id=m["repo_id"],
        revision=m["revision"],
        local_dir=str(root),
        allow_patterns=[x["path"] for x in needed],
        max_workers=workers,
    )
    print("DOWNLOAD_END", datetime.now(timezone.utc).isoformat(),
          f"seconds={round(time.monotonic()-t0,2)} path={path}", flush=True)
    t0 = time.monotonic()
    result = verify_files(root, include_preprocess=include_preprocess, verify_sha=True)
    print("MODEL_SHA256_VERIFY", json.dumps(result), flush=True)
    print("MODEL_VERIFY_SECONDS", round(time.monotonic()-t0,2), flush=True)
    if not result["complete"]:
        raise RuntimeErrorContract("Models failed checksum/size verification")
    return result


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,codec_type,width,height,avg_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=45
    )
    videos = json.loads(result.stdout).get("streams", [])
    if len(videos) != 1:
        raise RuntimeErrorContract(f"No single video stream in {path}")
    return videos[0]


def preflight_job(job_dir: Path, *, decode: bool = False):
    job_dir = job_dir.resolve()
    if not job_dir.is_dir():
        raise RuntimeErrorContract(f"Missing job directory: {job_dir}")
    frame_map = job_dir / "job.json"
    if not frame_map.is_file():
        raise RuntimeErrorContract(f"Missing normalized temporal metadata: {frame_map}")
    details = json.loads(frame_map.read_text())
    if details.get("mode") != "official-wan22-animate-replace":
        raise RuntimeErrorContract("Incorrect run mode")
    if details.get("source_fps") != "24000/1001":
        raise RuntimeErrorContract("Original Premiere FPS mismatch")
    if details.get("generation_fps") != 30 or details.get("generation_frames") != SAMPLE_FRAMES:
        raise RuntimeErrorContract("Wan frame/FPS contract mismatch")
    if details.get("motion_frames", 0) <= 0 or details.get("motion_frames", 0) > SAMPLE_FRAMES:
        raise RuntimeErrorContract("Invalid motion frame count")
    if details.get("source_frame_count", 0) <= 0:
        raise RuntimeErrorContract("Missing original source frame count")
    index = details.get("condition_frame_to_source_local_frame")
    if not isinstance(index, list) or len(index) != SAMPLE_FRAMES:
        raise RuntimeErrorContract("Frame-to-original-source mapping missing or incomplete")
    if not all(isinstance(x, int) and 0 <= x < details["source_frame_count"] for x in index):
        raise RuntimeErrorContract("Invalid original-frame index")
    if any(x > y for x, y in zip(index, index[1:])):
        raise RuntimeErrorContract("Source mapping must be monotonic")
    if index[0] != 0 or index[-1] != details["source_frame_count"] - 1:
        raise RuntimeErrorContract("First/final source frame boundaries not preserved")

    video_spec = {}
    for name in ASSETS:
        p = job_dir / name
        if not p.is_file():
            raise RuntimeErrorContract(f"Missing Wan condition: {p}")
        if name.endswith(".mp4"):
            info = ffprobe(p)
            if (Fraction(info["avg_frame_rate"]) != SAMPLE_FPS or
                    int(info["nb_read_frames"]) != SAMPLE_FRAMES):
                raise RuntimeErrorContract(f"{name}: require 77 frames at 30fps, got {info}")
            dimensions = (info["width"], info["height"])
            if dimensions != (FACE_SIZE if name == "src_face.mp4" else SIZE):
                raise RuntimeErrorContract(f"{name}: wrong dimensions {dimensions}")
            video_spec[name] = info
            if decode:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
                    check=True, timeout=100, capture_output=True
                )
        elif name == "src_ref.png":
            from PIL import Image
            with Image.open(p) as im:
                im.verify()
            from PIL import Image as PILImage
            with PILImage.open(p) as im:
                if im.width < 256 or im.height < 256:
                    raise RuntimeErrorContract("Reference photo is too small")
    print("PREFLIGHT_PASS", job_dir, "steps=20, frames=77, fps=30, video_sizes",
          {k: (v["width"], v["height"]) for k, v in video_spec.items()}, flush=True)
    return details


def smoke(*, gpu: bool = False):
    print(f"SMOKE_START time={datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"python={sys.version.split()[0]}", flush=True)
    assert sys.version_info >= (3, 10)
    if not UPSTREAM.joinpath("generate.py").is_file():
        raise RuntimeErrorContract(f"Official Wan source is absent: {UPSTREAM}")
    import torch
    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
          f"cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.__version__.startswith("2.5.1"):
        raise RuntimeErrorContract("Unexpected pinned torch version")
    if not str(torch.version.cuda).startswith("12.4"):
        raise RuntimeErrorContract("Unexpected CUDA runtime; requires CUDA 12.4.x")
    # Preserve the RunPod image's startup/Jupyter/SSH services.
    if not Path("/start.sh").is_file() and gpu:
        raise RuntimeErrorContract("Base image is missing RunPod /start.sh")
    sys.path.insert(0, str(UPSTREAM))
    # The official Wan source evaluates torch.cuda.current_device() as a
    # default argument while importing wan.modules.t5.T5EncoderModel. GitHub
    # Actions image builders have no NVIDIA driver. Patch ONLY that import-
    # time default to device 0 for a CPU import smoke; real inference still
    # uses actual torch.cuda.current_device on an attached GPU.
    if not gpu and not torch.cuda.is_available():
        from unittest.mock import patch
        print("CPU_IMPORT_ONLY: temporarily stub CUDA device index during import", flush=True)
        with patch.object(torch.cuda, "current_device", return_value=0):
            from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
            from wan.animate import WanAnimate
            from wan.modules.animate.preprocess.pose2d import Pose2d
    else:
        from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
        from wan.animate import WanAnimate
        from wan.modules.animate.preprocess.pose2d import Pose2d
    assert "animate-14B" in WAN_CONFIGS
    assert WAN_CONFIGS["animate-14B"].sample_steps == 20
    assert WAN_CONFIGS["animate-14B"].sample_fps == 30
    print("OFFICIAL_IMPORTS_PASS", flush=True)
    import flash_attn
    print("FLASH_ATTN_PASS", getattr(flash_attn, "__version__", "unknown"), flush=True)
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeErrorContract("No GPU provided; did you forget to launch a Pod?")
        properties = torch.cuda.get_device_properties(0)
        print("GPU", properties.name, properties.total_memory, flush=True)
        if properties.total_memory < 75 * (1024**3):
            raise RuntimeErrorContract("Full official Animate 14B requires a tested 80GB-class GPU")
    print("SMOKE_PASS", flush=True)


def run(job_dir: Path, models: Path, out: Path, steps: int, seed: int,
        *, actually_run: bool = False) -> list[str]:
    if steps < 1 or steps > 50:
        raise RuntimeErrorContract("Invalid sample step count; 1..50")
    if seed < 0:
        raise RuntimeErrorContract("Seed must be deterministic/non-negative")
    preflight_job(job_dir, decode=True)
    m = verify_files(models, verify_sha=False)
    if not m["complete"]:
        raise RuntimeErrorContract(f"Missing checkpoint files: {m['errors'][:8]}")
    if not actually_run:
        print("DRY_RUN_ONLY: no GPU inference or writing will occur", flush=True)
    if actually_run:
        smoke(gpu=True)
    out = out.resolve()
    if actually_run:
        if out.exists():
            raise RuntimeErrorContract("Refusing to overwrite an existing generated video")
        out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(UPSTREAM / "generate.py"),
        "--task", "animate-14B",
        "--size", "1280*720",  # Official parser accepts this; real input size comes from src_pose.
        "--ckpt_dir", str(models.resolve()),
        "--src_root_path", str(job_dir.resolve()),
        "--frame_num", str(SAMPLE_FRAMES),
        "--sample_steps", str(steps),
        "--refert_num", "1",
        "--base_seed", str(seed),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--replace_flag",
        "--use_relighting_lora",
        "--save_file", str(out),
    ]
    print("OFFICIAL_COMMAND", " ".join(command), flush=True)
    if actually_run:
        print("INFERENCE_START", datetime.now(timezone.utc).isoformat(), flush=True)
        begin = time.monotonic()
        subprocess.run(command, cwd=UPSTREAM, check=True)
        took = round(time.monotonic() - begin, 2)
        print("INFERENCE_END", datetime.now(timezone.utc).isoformat(),
              f"seconds={took}", flush=True)
        data = ffprobe(out)
        if int(data["nb_read_frames"]) != SAMPLE_FRAMES or Fraction(data["avg_frame_rate"]) != SAMPLE_FPS:
            raise RuntimeErrorContract("Generated video does not decode to expected 77 frames @ 30fps")
        print("RESULT_SHA256", digest(out), flush=True)
    return command


def main():
    parser = argparse.ArgumentParser(description="Pinned official Wan2.2 Animate runtime for disposable GPUs")
    cmds = parser.add_subparsers(dest="command", required=True)
    plan = cmds.add_parser("plan", help="Show exact pinned checkpoint requirements, no download")
    plan.add_argument("--with-preprocess", action="store_true")
    check = cmds.add_parser("models-verify", help="Verify checkpoint files")
    check.add_argument("--root", type=Path, default=DEFAULT_MODELS)
    check.add_argument("--sha256", action="store_true")
    check.add_argument("--with-preprocess", action="store_true")
    dl = cmds.add_parser("models-download", help="EXPLICIT download to disposable disk only")
    dl.add_argument("--root", type=Path, default=DEFAULT_MODELS)
    dl.add_argument("--with-preprocess", action="store_true")
    dl.add_argument("--workers", type=int, default=8)
    st = cmds.add_parser("smoke", help="Import full official source in CPU-only mode or inspect GPU")
    st.add_argument("--gpu", action="store_true")
    pf = cmds.add_parser("preflight", help="Verify all prepared Wan condition videos (no inference)")
    pf.add_argument("--job-dir", type=Path, required=True)
    pf.add_argument("--decode", action="store_true")
    infer = cmds.add_parser("infer", help="Official Wan replacement; requires running authorized Pod")
    infer.add_argument("--job-dir", type=Path, required=True)
    infer.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--steps", type=int, default=20)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            print_plan(args.with_preprocess)
        elif args.command == "models-verify":
            result = verify_files(args.root, include_preprocess=args.with_preprocess,
                                  verify_sha=args.sha256)
            print(json.dumps(result, indent=2))
            if not result["complete"]:
                return 1
        elif args.command == "models-download":
            if not 1 <= args.workers <= 16:
                parser.error("--workers must be 1..16")
            download_models(args.root, args.with_preprocess, args.workers)
        elif args.command == "smoke":
            smoke(gpu=args.gpu)
        elif args.command == "preflight":
            preflight_job(args.job_dir, decode=args.decode)
        elif args.command == "infer":
            run(args.job_dir, args.models, args.output, args.steps, args.seed,
                actually_run=not args.dry_run)
        return 0
    except (RuntimeErrorContract, subprocess.CalledProcessError, OSError,
            ValueError, AssertionError) as error:
        print("WAN22_ERROR", str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
