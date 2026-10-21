"""Serve baidu/Unlimited-OCR with vLLM on Modal, behind a bearer token.

The server exposes an OpenAI-compatible ``/v1`` endpoint for ``app/ocr_remote.py``. It runs on one
L4 (24 GB; the 3B BF16 model needs about 8 GB, the rest is KV cache for page crops),
scales to zero after a few idle minutes and caches the weights in a Modal Volume.

One-time setup (the token never goes in the repository)::

    modal secret create unlimited-ocr-auth OCR_API_KEY="$(openssl rand -base64 32)"
    modal deploy evals/modal_unlimited_ocr.py

Then point the app at ``https://<workspace>--unlimited-ocr-serve.modal.run/v1`` with
``PDFCHAT_OCR_REMOTE_API_KEY`` set to the same token. Stop it with
``modal app stop unlimited-ocr`` when you are done.
"""

from __future__ import annotations

import subprocess

import modal

MODEL = "baidu/Unlimited-OCR"
# The Hub commit of 2026-10-14 renamed model_type to "unlimited_ocr", which the release image's
# native config (keyed "unlimited-ocr") no longer matches; the image then falls back to the repo's
# remote code, which needs an older transformers than vLLM allows. This is the last good commit.
REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
VLLM_PORT = 8000
GPU = "L4"
SCALEDOWN_S = 180

app = modal.App("unlimited-ocr")

image = modal.Image.from_registry("vllm/vllm-openai:unlimited-ocr", add_python="3.12").entrypoint(
    []
)
weights = modal.Volume.from_name("unlimited-ocr-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": weights},
    secrets=[modal.Secret.from_name("unlimited-ocr-auth")],
    scaledown_window=SCALEDOWN_S,
    timeout=60 * 60,
    max_containers=1,
    # Model load can take several minutes on a cold start.
    startup_timeout=20 * 60,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * 60)
def serve() -> None:
    import os

    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--host",
        "0.0.0.0",  # noqa: S104 - inside the Modal container; Modal fronts it with TLS
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["OCR_API_KEY"],
        "--trust-remote-code",
        "--logits_processors",
        "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
    ]
    # Popen, not run: web_server waits for the port while this keeps serving.
    subprocess.Popen(cmd)  # noqa: S603 - fixed argv, no user input
