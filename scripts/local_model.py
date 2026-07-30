"""
local_model.py — thin wrapper around llama-cpp-python for local H3 experiments.

Supports:
    - Phi-3 Mini 4K Instruct (GGUF)
    - Qwen2.5 3B Instruct (GGUF)

Works with:
    llama-cpp-python >= 0.3.34

Notes
-----
Modern GGUF files (including Bartowski's) usually embed their own chat
template. We therefore DO NOT hardcode a chat_format for Phi-3.

Priority used by llama-cpp-python:

1. chat_handler
2. chat_format
3. GGUF tokenizer.chat_template
4. llama-2 fallback
"""

from pathlib import Path

from llama_cpp import Llama

ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIGS = {
    "phi3-mini": {
        "path": ROOT / "models" / "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        #NO chat_format.
        #Bartowski GGUF embeds the correct template.
        "n_ctx": 4096,
    },
    "qwen2.5-3b": {
        "path": ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "chat_format": "chatml",
        "n_ctx": 4096,
    },
}

_loaded = {}


def get_model(model_key: str, n_threads: int | None = None) -> Llama:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    if model_key in _loaded:
        return _loaded[model_key]

    cfg = MODEL_CONFIGS[model_key]

    model_path = cfg["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(f"\nLoading {model_key}...")
    print(f"Model: {model_path.name}")

    kwargs = {
        "model_path": str(model_path),
        "n_ctx": cfg.get("n_ctx", 4096),
        "n_threads": n_threads,
        "verbose": False,  
    }

    if "chat_format" in cfg:
        kwargs["chat_format"] = cfg["chat_format"]

    llm = Llama(**kwargs)

    _loaded[model_key] = llm

    print(f"{model_key} loaded successfully.\n")

    return llm


def call_model(
    model_key: str,
    prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.0,
) -> str:
    llm = get_model(model_key)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "phi3-mini"

    print("=" * 60)
    print(f"Testing: {model}")
    print("=" * 60)

    output = call_model(
        model,
        "What is 2 + 2? Answer with only the number."
    )

    print("\nModel output:")
    print("-" * 60)
    print(output)
    print("-" * 60)