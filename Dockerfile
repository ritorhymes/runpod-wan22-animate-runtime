# Reproducible, code-only official Wan2.2-Animate-14B runtime.
# CUDA/Torch pinned to RunPod base. NO weights, footage, reference photos, or secrets.
FROM runpod/pytorch:0.7.0-cu1241-torch251-ubuntu2204@sha256:3bdc5341e5261bb4fbe009498eb437717ab53e92bdbd13e957c5927bb8fca910
ARG WAN_SOURCE_REV=1ea34ff48f87168174e12956e200b1d908b1c5ff
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    WAN22_UPSTREAM=/opt/wan2.2 \
    WAN22_MODELS=/workspace/models/Wan2.2-Animate-14B \
    PYTHONPATH=/opt/wan2.2
RUN test -f /start.sh || (echo "RunPod SSH/Jupyter startup script missing from chosen image" >&2; exit 2)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl ffmpeg libgomp1 libglib2.0-0 libcairo2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-runtime.txt /opt/wan22/requirements-runtime.txt
RUN python3 -m pip install --no-cache-dir -r /opt/wan22/requirements-runtime.txt \
    && python3 -c "import torch,numpy; assert torch.__version__.startswith('2.5.1'); assert numpy.__version__=='1.26.4'; print('Torch:',torch.__version__,'numpy:',numpy.__version__)"
COPY scripts/install-flash-attn.sh /opt/wan22/scripts/install-flash-attn.sh
RUN chmod 755 /opt/wan22/scripts/install-flash-attn.sh \
    && /opt/wan22/scripts/install-flash-attn.sh
# Fetch the exact official Wan source during the FREE GitHub Actions build,
# not during expensive Pod startup. Never accept a moving main ref.
RUN git clone --filter=blob:none --no-checkout https://github.com/Wan-Video/Wan2.2.git /opt/wan2.2 \
    && git -C /opt/wan2.2 checkout --detach "$WAN_SOURCE_REV" \
    && test "$(git -C /opt/wan2.2 rev-parse HEAD)" = "$WAN_SOURCE_REV" \
    && rm -rf /opt/wan2.2/.git
COPY model-manifest.json /opt/wan22/model-manifest.json
COPY scripts/runtime.py /opt/wan22/scripts/runtime.py
COPY scripts/verify-deps.py /opt/wan22/scripts/verify-deps.py
COPY scripts/wan22-runtime /opt/wan22/scripts/wan22-runtime
RUN chmod 755 /opt/wan22/scripts/wan22-runtime /opt/wan22/scripts/runtime.py \
    && ln -s /opt/wan22/scripts/wan22-runtime /usr/local/bin/wan22-runtime \
    && python3 -m compileall -q /opt/wan22/scripts \
    && wan22-runtime smoke \
    && wan22-runtime plan \
    && python3 /opt/wan22/scripts/verify-deps.py
WORKDIR /workspace
# DO NOT override ENTRYPOINT or CMD. Inherit RunPod's SSH and Jupyter startup.
