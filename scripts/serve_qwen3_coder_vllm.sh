#!/usr/bin/env bash
#
# serve_qwen3_coder_vllm.sh
# -------------------------
# Launch a private, OpenAI-compatible inference server for
# Qwen/Qwen3-Coder-30B-A3B-Instruct using vLLM, listening on port 8000.
#
# This is a ~30B Mixture-of-Experts checkpoint and it needs the full 80GB GPU,
# so YOU CAN ONLY RUN ONE BIG MODEL AT A TIME.
# Stop any other large model before starting this one (Ctrl-C its process, or free
# the GPU) — otherwise vLLM will fail to allocate memory.
#
# Once running you can query it via curl or any OpenAI-compatible client
# (including PatchProof's --llm mode). Point PatchProof at it with:
#   --llm-provider vllm --llm-base-url http://<host>:8000/v1 \
#   --llm-model Qwen3-Coder-30B-A3B-Instruct
# See scripts/test_vllm_server.sh for a quick smoke test.
#
# Usage:
#   bash scripts/serve_qwen3_coder_vllm.sh
#
# Common overrides (export before running, or prefix the command):
#   PORT=8000                       # port to listen on
#   HOST=0.0.0.0                    # bind address (0.0.0.0 = reachable from other hosts)
#   MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
#   SERVED_MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct  # name clients pass as "model"
#   TENSOR_PARALLEL_SIZE=<auto>     # defaults to the number of visible GPUs
#   MAX_MODEL_LEN=32768             # context window (model supports up to 262144)
#   GPU_MEMORY_UTILIZATION=0.90     # fraction of GPU memory vLLM may use
#   API_KEY=<unset>                 # if set, clients must send this bearer token
#   DTYPE=auto                      # auto | bfloat16 | float16
#   ENABLE_TOOLS=0                  # set to 1 to enable OpenAI tool/function calling
#   EXTRA_ARGS=""                   # any additional raw `vllm serve` flags
#   SKIP_INSTALL=0                  # set to 1 to skip the pip install step
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
ENABLE_TOOLS="${ENABLE_TOOLS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

log() { printf '[serve_qwen3_coder_vllm] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Detect GPUs and pick a sensible tensor-parallel size.
# ---------------------------------------------------------------------------
detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

GPU_COUNT="$(detect_gpu_count)"
if [ "${GPU_COUNT}" -eq 0 ]; then
  log "WARNING: no NVIDIA GPU detected via nvidia-smi."
  log "Qwen3-Coder-30B-A3B is a Mixture-of-Experts model (30B total params); it needs"
  log "a CUDA GPU with enough memory (e.g. a single 80GB card, or 2x 48GB with"
  log "TENSOR_PARALLEL_SIZE=2). vLLM will likely fail to start without a GPU."
fi

if [ -z "${TENSOR_PARALLEL_SIZE:-}" ]; then
  if [ "${GPU_COUNT}" -ge 1 ]; then
    TENSOR_PARALLEL_SIZE="${GPU_COUNT}"
  else
    TENSOR_PARALLEL_SIZE=1
  fi
fi

# ---------------------------------------------------------------------------
# Guard: only one ~30B model fits on a single 80GB GPU at a time.
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  USED_MEM_MB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
  if [ -n "${USED_MEM_MB:-}" ] && [ "${USED_MEM_MB}" -gt 5000 ]; then
    log "WARNING: GPU already has ${USED_MEM_MB} MiB in use. Another large model may"
    log "still be running. Stop it first — you cannot run more than one ~30B model on"
    log "a single 80GB GPU at the same time."
  fi
fi

# ---------------------------------------------------------------------------
# Install vLLM if it is not already importable.
# ---------------------------------------------------------------------------
if [ "${SKIP_INSTALL}" != "1" ]; then
  if ! python3 -c "import vllm" >/dev/null 2>&1; then
    log "vLLM not found; installing (this pulls in PyTorch + CUDA wheels and can take a while)..."
    python3 -m pip install --upgrade pip
    # Qwen3-Coder-30B-A3B-Instruct requires a recent vLLM (>=0.9.x) and transformers.
    python3 -m pip install "vllm>=0.9.1"
  else
    log "vLLM already installed: $(python3 -c 'import vllm; print(vllm.__version__)')"
  fi
fi

# ---------------------------------------------------------------------------
# Assemble the vllm serve command.
# ---------------------------------------------------------------------------
ARGS=(
  serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --trust-remote-code
)

if [ -n "${API_KEY:-}" ]; then
  ARGS+=(--api-key "${API_KEY}")
  log "API key authentication is ENABLED (clients must send: Authorization: Bearer <API_KEY>)."
fi

if [ "${ENABLE_TOOLS}" = "1" ]; then
  # Qwen3-Coder ships its own tool-call template/parser.
  ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
  log "OpenAI tool/function calling is ENABLED (qwen3_coder parser)."
fi

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA_ARGS})
fi

# ---------------------------------------------------------------------------
# Launch.
# ---------------------------------------------------------------------------
log "Starting vLLM OpenAI-compatible server"
log "  model                  = ${MODEL}"
log "  served name            = ${SERVED_MODEL_NAME}"
log "  endpoint               = http://${HOST}:${PORT}/v1"
log "  tensor parallel size   = ${TENSOR_PARALLEL_SIZE} (GPUs detected: ${GPU_COUNT})"
log "  max model len          = ${MAX_MODEL_LEN}"
log "  gpu memory utilization = ${GPU_MEMORY_UTILIZATION}"
log ""
log "Once you see 'Uvicorn running on http://${HOST}:${PORT}', the server is ready."
log "Smoke test it with: bash scripts/test_vllm_server.sh"
log "Query from PatchProof with: --llm-provider vllm --llm-model ${SERVED_MODEL_NAME}"
log ""
log "Running: vllm ${ARGS[*]}"

exec vllm "${ARGS[@]}"
