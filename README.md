# Official Wan 2.2 Animate 14B — disposable GPU runtime

This **public code-only repository** packages the official full-precision Wan 2.2 character-replacement inference environment. It NEVER contains user video, references, masks, private tokens, SSH keys, or large model weights.

Official source: https://github.com/Wan-Video/Wan2.2 pinned to commit 1ea34ff48f87168174e12956e200b1d908b1c5ff.
Official model: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B pinned to revision cb93a225fbaf1ca100f54e79da8f994995b689b3.
Runtime: Torch 2.5.1, CUDA 12.4, FlashAttention 2.7.4.post1 verified against official prebuilt-wheel SHA256.

## Free GitHub Actions build

The GitHub workflow builds a code-only GHCR image and runs a CPU-only official import smoke test. It requires the GitHub Container Registry package to be **PUBLIC** before uploading large layers. No RunPod Pod or other billable GPU is used by the workflow.

Published images (after the workflow succeeds):
- ghcr.io/ritorhymes/runpod-wan22-animate-runtime:main
- ghcr.io/ritorhymes/runpod-wan22-animate-runtime:sha-FULL_COMMIT_SHA

The base image inherits RunPod's startup command, SSH and notebook services. No weights are baked into its layers.

## Runtime commands on an already-running authorized Pod

    wan22-runtime smoke --gpu
    wan22-runtime plan
    wan22-runtime models-download
    wan22-runtime models-verify --root /workspace/models/Wan2.2-Animate-14B --sha256
    wan22-runtime preflight --job-dir /workspace/wan22/jobs/s01e03-clip03-shot19 --decode
    wan22-runtime infer --job-dir /workspace/wan22/jobs/s01e03-clip03-shot19 --output /workspace/wan22/output/s01e03-clip03-shot19-wan22-replaced.mp4 --steps 20 --seed 42

**T5 placement:** `infer` defaults to `--t5-placement auto`, which selects the official GPU T5 path on a tested 80GB-class GPU. There is **no automatic CPU fallback**: an undersized GPU fails preflight. Explicit `--t5-placement gpu` also chooses GPU; `--t5-placement cpu` is available only as an intentional override and may take a very long time on CPUs lacking native BF16 acceleration. The BF16 weights, reference conditioning, seed, and diffusion settings are unchanged. Dry-run planning does not access CUDA. This behavior was verified on an A100 80GB after CPU T5 stalled on an AMD EPYC 7742 host.

The explicit model download fetches only the pinned official generation files (54,100,856,515 bytes) onto **disposable Pod disk**. It never provisions network volumes. The generated video is returned to the local controller.

The five official conditioning assets are src_ref.png, src_face.mp4 (512x512), src_pose.mp4, src_mask.mp4 and src_bg.mp4 (832x480, same original-frame mapping). The private controller generates them locally from the original Premiere cuts, reference image and user-approved SAM2 masks. Do not commit that private content.

## Current first-test specifics

- Test Shot 19: 40 source frames at 24000/1001 FPS, 832x480.
- Converted to 53 Wan frames at 30 FPS (4n+1), preserving the original source motion and holding the last frame for minimal padding.
- Official sample count: 20 diffusion steps, one reference, fixed seed 42.
- The user-selected new reference photo is used, not an older one.
- The first 53-frame paid acceptance run completed on an A100 80GB using GPU T5, with the approved reference and official BF16 inference. Larger conditioning windows still need per-shot memory and output checks.

## Safety

- No Pod creation functionality exists in this public repository.
- Paid GPU launch REQUIRES separate, explicit user authorization.
- Use only a single disposable Pod with no network volumes or permanent storage.
- Obtain authorization for the requested shot/batch, preserve logs and input integrity, and never expand the authorized scope. The initial Shot 19 acceptance is complete; the separately authorized ten-shot batch runs through the private controller.
- User decides when to delete the Pod. The private controller's rgpu kill command is the available emergency kill switch.

## Tests

    python3 -m unittest discover -s tests -v
    python3 scripts/runtime.py plan

These commands do not download weights, start a Pod or charge money.

## Known limitations

Official Wan imports call torch.cuda.current_device at module-import time. CPU GitHub Actions smoke tests temporarily substitute the device index *only during import*. Real GPU checks will run without that substitution on the paid Pod. The pose output for a head-only actor may be sparse, and video generation may alter surrounding pixels despite exact conditioning masks. Each paid result should be reviewed. GPU T5 memory headroom for 80GB GPUs may be insufficient for longer jobs; abort rather than silently lowering model precision.
