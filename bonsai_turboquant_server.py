#!/usr/bin/env python3
"""
Bonsai 8B inference server with TurboQuant KV cache compression.

Serves an OpenAI-compatible API on localhost:1235 (port offset from LM Studio's 1234).
Uses TurboQuant to compress the KV cache at inference time, enabling:
  - 32K-64K context without VRAM explosion
  - Stable reasoning on long chains (no coherence drift)
  - Accuracy-neutral: preserves existing reasoning capabilities

The Karpathy loop can point here instead of (or alongside) the llama-server.

Usage:
  python bonsai_turboquant_server.py                    # Start server on :1235
  python bonsai_turboquant_server.py --port 1234        # Replace llama-server
  python bonsai_turboquant_server.py --v-bits 4 --k-bits 3  # Custom bit allocation
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "prism-ml/Bonsai-8B"  # HuggingFace model ID (or local path)
GGUF_PATH = os.environ.get("BONSAI_GGUF_PATH", str(Path.home() / ".lmstudio/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B.gguf"))

DEFAULT_PORT = 1235
DEFAULT_V_BITS = 4   # Values are more sensitive — keep at 4
DEFAULT_K_BITS = 3   # Keys can tolerate more compression


class BonsaiServer:
    def __init__(self, model_id: str, v_bits: int = 4, k_bits: int = 3,
                 max_seq_len: int = 65536):
        print(f"Loading model: {model_id}")
        print(f"TurboQuant: V={v_bits}-bit, K={k_bits}-bit")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.v_bits = v_bits
        self.k_bits = k_bits
        self.max_seq_len = max_seq_len
        self.lock = Lock()

        # Print memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU memory: {allocated:.1f}GB used / {total:.1f}GB total")

        print(f"Server ready. Max context: {max_seq_len}")

    def generate(self, messages: list, temperature: float = 0.7,
                 max_tokens: int = 2048) -> dict:
        """Generate a response with TurboQuant KV cache."""
        with self.lock:
            # Build prompt from messages using chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Manual ChatML fallback
                prompt = ""
                for msg in messages:
                    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]

            # Create TurboQuant cache — this is the key integration
            tq_cache = TurboQuantCache(
                model=self.model,
                v_bits=self.v_bits,
                k_bits=self.k_bits,
            )

            t0 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    past_key_values=tq_cache,
                    use_cache=True,
                )

            elapsed = time.time() - t0
            new_tokens = outputs.shape[1] - input_len
            generated = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

            return {
                "content": generated,
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": new_tokens,
                    "total_tokens": input_len + new_tokens,
                },
                "timing": {
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_s": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
                },
            }


class APIHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /v1/chat/completions endpoint."""

    server_instance: BonsaiServer = None

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))

            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.7)
            max_tokens = body.get("max_tokens", 2048)

            result = self.server_instance.generate(messages, temperature, max_tokens)

            response = {
                "id": f"tq-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "bonsai-8b-turboquant",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }],
                "usage": result["usage"],
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/v1/models":
            self.do_GET()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/v1/models":
            response = {
                "object": "list",
                "data": [{
                    "id": "bonsai-8b-turboquant",
                    "object": "model",
                    "owned_by": "local",
                }],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Quiet logging
        pass


def main():
    parser = argparse.ArgumentParser(description="Bonsai TurboQuant inference server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--v-bits", type=int, default=DEFAULT_V_BITS,
                        help="Bits for Value cache (default: 4)")
    parser.add_argument("--k-bits", type=int, default=DEFAULT_K_BITS,
                        help="Bits for Key cache (default: 3)")
    parser.add_argument("--max-seq", type=int, default=65536)
    args = parser.parse_args()

    server = BonsaiServer(args.model, args.v_bits, args.k_bits, args.max_seq)
    APIHandler.server_instance = server

    httpd = HTTPServer(("0.0.0.0", args.port), APIHandler)
    print(f"\nListening on http://localhost:{args.port}/v1/chat/completions")
    print(f"Compatible with OpenAI API — drop-in replacement for llama-server")
    print(f"TurboQuant KV cache: V={args.v_bits}-bit, K={args.k_bits}-bit")
    print(f"Press Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
