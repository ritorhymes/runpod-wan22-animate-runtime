#!/usr/bin/env bash
# Pull the exact official FlashAttention wheel for Torch 2.5 CUDA12, no CUDA compilation.
set -euo pipefail
PY_TAG="$(python3 -c 'import sys; print("cp%d%d" % (sys.version_info.major, sys.version_info.minor))')"
case "$PY_TAG" in
  cp310) SHA256="b267f80a08e516292cdd748056a2178a45b8abedf7fca123292eb17c21c8c87c";;
  cp311) SHA256="d944fc7d2f962bce83fc4708c2fc0c21eaf8255962a0b350ae919362a51b7ef2";;
  cp312) SHA256="a496f383c843ed0cc6e01302556a0947f56828f1ec2b54e5297c4c7eaae39357";;
  *) echo >&2 "Unsupported FlashAttention wheel Python version $PY_TAG"; exit 2;;
esac
python3 - <<'PY'
import torch
print("Torch",torch.__version__,"CUDA",torch.version.cuda,"ABI",torch._C._GLIBCXX_USE_CXX11_ABI)
assert torch.__version__.startswith("2.5.1"), "Prebuilt FlashAttention requires torch 2.5.1"
assert str(torch.version.cuda).startswith("12.4"), "CUDA 12.4 required"
assert not torch._C._GLIBCXX_USE_CXX11_ABI, "Wheel requires cxx11abiFALSE"
PY
NAME="flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-$PY_TAG-$PY_TAG-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-$PY_TAG-$PY_TAG-linux_x86_64.whl"
echo "Installing checked official wheel $NAME"
curl -fLsS --retry 3 --connect-timeout 15 --max-time 300 "$URL" -o "/tmp/$NAME"
printf '%s  %s\n' "$SHA256" "/tmp/$NAME" | sha256sum --check --status
python3 -m pip install --no-deps --no-cache-dir "/tmp/$NAME"
rm -f "/tmp/$NAME"
python3 - <<'PY'
import flash_attn
print("FlashAttention import OK",flash_attn.__version__)
assert flash_attn.__version__ == "2.7.4.post1"
PY
