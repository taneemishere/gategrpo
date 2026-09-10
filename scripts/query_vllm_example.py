#!/usr/bin/env python3
"""Minimal OpenAI-compatible client for the local vLLM Qwen3 server.

Uses only the standard library (urllib), mirroring gategrpo/llm.py, so it
can run anywhere without extra dependencies. This is the same request shape
GateGRPO's --llm mode will use once the repo is pointed at the local server.

Examples:
    python3 scripts/query_vllm_example.py
    python3 scripts/query_vllm_example.py --prompt "Write a haiku about patches"

Environment overrides:
    VLLM_BASE_URL          default http://localhost:8000/v1
    VLLM_MODEL             default Qwen3-Coder-30B-A3B-Instruct
    VLLM_API_KEY           sent as a bearer token if set (matches server --api-key)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def chat(base_url: str, model: str, prompt: str, api_key: str | None) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the local vLLM Qwen3 server.")
    parser.add_argument(
        "--base-url",
        # default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL", "Qwen3-Coder-30B-A3B-Instruct"),
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: GateGRPO vLLM server OK",
    )
    args = parser.parse_args()
    api_key = os.environ.get("VLLM_API_KEY") or None

    try:
        result = chat(args.base_url, args.model, args.prompt, api_key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} error from {args.base_url}: {body}")
        return 1
    except urllib.error.URLError as exc:
        print(f"Could not reach {args.base_url}: {exc.reason}")
        print("Is the server running? Start it with: bash scripts/serve_qwen3_coder_vllm.sh")
        return 1

    content = result["choices"][0]["message"]["content"]
    print("Model:", result.get("model"))
    print("Response:", content)
    usage = result.get("usage")
    if usage:
        print("Usage:", json.dumps(usage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
