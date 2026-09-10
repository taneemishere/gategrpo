#!/usr/bin/env bash
#
# test_vllm_server.sh
# -------------------
# Quick smoke test for the vLLM server started by serve_qwen3_coder_vllm.sh.
# Verifies the server is up, lists the served model, and sends one chat
# completion request through the OpenAI-compatible API.
#
# Usage:
#   bash scripts/test_vllm_server.sh
#
# Overrides:
#   HOST=localhost
#   PORT=8000
#   SERVED_MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct
#   API_KEY=<unset>   # must match the server's --api-key if one was set
#
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
BASE_URL="http://${HOST}:${PORT}/v1"

AUTH_HEADER=()
if [ -n "${API_KEY:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")
fi

echo "==> 1. Listing models at ${BASE_URL}/models"
curl -sS "${AUTH_HEADER[@]}" "${BASE_URL}/models"
echo
echo

echo "==> 2. Sending a chat completion to model '${SERVED_MODEL_NAME}'"
curl -sS "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  "${BASE_URL}/chat/completions" \
  -d "$(cat <<JSON
{
  "model": "${SERVED_MODEL_NAME}",
  "messages": [
    {"role": "user", "content": "Reply with exactly: GateGRPO vLLM server OK"}
  ],
  "max_tokens": 32,
  "temperature": 0
}
JSON
)"
echo
echo
echo "==> If you see a JSON response with a 'choices' array above, the server works."
