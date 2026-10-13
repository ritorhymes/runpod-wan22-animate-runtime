#!/usr/bin/env python3
"""Validate the actual Wan runtime instead of trusting old Decord wheel metadata.

Official Decord 0.6.0 may report an unsupported-platform metadata warning with
pip check despite being importable. Test it by decoding real frames before
allowlisting ONLY that exact legacy warning. All other pip check errors fail.
"""
from __future__ import annotations
import importlib
import pathlib
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as d:
    video=pathlib.Path(d)/"decode-test.mp4"
    subprocess.run([
        "ffmpeg","-hide_banner","-nostdin","-loglevel","error","-y",
        "-f","lavfi","-i","color=c=red:s=64x64:r=30",
        "-frames:v","4","-c:v","libx264","-pix_fmt","yuv420p",
        str(video),
    ],check=True,timeout=45)
    import decord
    reader=decord.VideoReader(str(video),ctx=decord.cpu(0))
    if len(reader)!=4:
        raise RuntimeError("Official Decord did not decode all four frames")
    if reader[0].asnumpy().shape[:2] != (64,64):
        raise RuntimeError("Decord decoded wrong image dimensions")
    print("DECORD_NATIVE_CPU_VIDEO_DECODE_PASS",len(reader),flush=True)
for module in (
    "torch","torchvision","flash_attn","decord","librosa",
    "diffusers","peft","transformers","onnxruntime","pandas","matplotlib",
    "cv2","moviepy","huggingface_hub",
):
    package=importlib.import_module(module)
    print("DEPENDENCY_OK",module,getattr(package,"__version__","imported"),flush=True)
result=subprocess.run([sys.executable,"-m","pip","check"],
                      capture_output=True,text=True,timeout=30)
print("PIP_CHECK_OUTPUT",result.stdout.strip(),flush=True)
print(result.stderr.strip(),flush=True)
if result.returncode:
    notes=[line.strip() for line in result.stdout.splitlines() if line.strip()]
    allowed={"decord 0.6.0 is not supported on this platform"}
    unexpected=set(notes)-allowed
    if unexpected or not set(notes):
        raise RuntimeError(f"Unverified dependency conflict(s): {sorted(unexpected or notes)}")
    print("KNOWN_DECORD_METADATA_WARNING_ACCEPTED_AFTER_REAL_VIDEO_DECODE",flush=True)
print("PINNED_WAN_DEPENDENCY_VERIFICATION_PASS",flush=True)
