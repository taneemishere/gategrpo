# Local LLM serving for PatchProof

These scripts stand up a **private, OpenAI-compatible inference server** for
`Qwen/Qwen3-Coder-30B-A3B-Instruct` using [vLLM], listening on **port 8000**.
This is the first step toward live-LLM evaluation across the full task suite:
the controlled benchmark stays deterministic, while a self-hosted model lets us
run real generation without a third-party API.

| Script | Purpose |
| --- | --- |
| `serve_qwen3_coder_vllm.sh` | Install vLLM (if needed) and launch the Qwen3-Coder-30B-A3B-Instruct server on port 8000. |
| `test_vllm_server.sh` | curl smoke test: list models + one chat completion. |
| `query_vllm_example.py` | Zero-dependency Python client (stdlib `urllib`). |

## Start the server

```bash
bash scripts/serve_qwen3_coder_vllm.sh
```

The server exposes an OpenAI-compatible API at `http://<host>:8000/v1`. When you
see `Uvicorn running on http://0.0.0.0:8000` it is ready.

`Qwen3-Coder-30B-A3B-Instruct` is a Mixture-of-Experts model (30B total
parameters, ~3B active) with native 256K context.
It needs a CUDA GPU with enough memory — e.g. a single 80GB card, or set
`TENSOR_PARALLEL_SIZE=2` to shard across two smaller cards. The script
auto-detects the visible GPU count for tensor parallelism.

### Useful overrides

```bash
PORT=8000 \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.90 \
TENSOR_PARALLEL_SIZE=2 \
bash scripts/serve_qwen3_coder_vllm.sh
```

Set `API_KEY=...` to require a bearer token, `ENABLE_TOOLS=1` for OpenAI
tool/function calling, or `EXTRA_ARGS="--quantization fp8"` to pass raw vLLM
flags. See the header of `serve_qwen3_coder_vllm.sh` for the full list.

## Test it

```bash
# from the same host
bash scripts/test_vllm_server.sh

# or with the Python client
python3 scripts/query_vllm_example.py
```

Querying from another machine works the same way — point the client at
`http://<server-host>:8000/v1`:

```bash
curl http://<server-host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Use it from PatchProof:

```bash
python3 -m patchproof demo --run-dir .patchproof_runs/demo --llm \
  --llm-provider vllm --llm-base-url http://localhost:8000/v1
```

Only one ~30B model fits on a single 80GB GPU at a time.

[vLLM]: https://docs.vllm.ai/
