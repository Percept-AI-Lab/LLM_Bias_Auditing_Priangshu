"""
00_setup.py
============
Windows Setup & Sanity Check for BanStereoSet Pipeline

Run this ONCE before any other script to:
  1. Install llama-cpp-python with CUDA (cuBLAS) support
  2. Download Qwen2.5-3B-Instruct Q4_K_M GGUF (~2.0 GB)
  3. Verify the model loads and generates correctly
  4. Create the outputs/ directory

WHY GGUF + llama-cpp-python?
------------------------------
bitsandbytes does not support Windows natively — its CUDA kernels are Linux-only.
llama-cpp-python uses the GGUF format which:
  • Works on Windows with CUDA (cuBLAS backend)
  • Uses ~2.0 GB VRAM for Qwen2.5-3B Q4_K_M (fits easily in 6 GB)
  • Requires no PyTorch CUDA setup
  • Is significantly faster on Windows than bitsandbytes alternatives

USAGE:
    uv run 00_setup.py
    # or
    python 00_setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_REPO   = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE   = "qwen2.5-3b-instruct-q4_k_m.gguf"   # ~2.0 GB, best quality/size tradeoff
LOCAL_MODEL  = Path("models") / MODEL_FILE
OUTPUTS_DIR  = Path("outputs")


def run(cmd: str, check=True):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def install_llama_cpp():
    """
    Install llama-cpp-python with CUDA (cuBLAS) support.
    This enables GPU offloading on the RTX 4050.
    """
    print("\n" + "="*60)
    print("STEP 1 — Installing llama-cpp-python (CUDA/cuBLAS build)")
    print("="*60)

    # Check if already installed
    try:
        import llama_cpp
        print("[OK] llama-cpp-python already installed.")
        return
    except ImportError:
        pass

    # Install with cuBLAS for NVIDIA GPU acceleration
    # CMAKE_ARGS tells the build to compile CUDA kernels
    env_cmd = 'set "CMAKE_ARGS=-DGGML_CUDA=on" && '
    pip_cmd = f'{env_cmd}pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --upgrade'

    result = run(pip_cmd, check=False)
    if result != 0:
        print("[WARN] cuBLAS wheel failed, trying pre-built CUDA 12.1 wheel...")
        # Direct pre-built wheel for Windows + CUDA 12.x
        wheel_cmd = (
            f'pip install llama-cpp-python '
            f'--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121'
        )
        result2 = run(wheel_cmd, check=False)
        if result2 != 0:
            print("[WARN] Pre-built wheel also failed. Installing CPU-only version.")
            print("       (will still work but slower — all computation on CPU)")
            run("pip install llama-cpp-python")

    # Verify
    try:
        import llama_cpp
        print(f"[OK] llama-cpp-python installed: {llama_cpp.__version__}")
    except ImportError:
        print("[ERROR] llama-cpp-python failed to install. Check your CUDA installation.")
        sys.exit(1)


def install_other_deps():
    print("\n" + "="*60)
    print("STEP 2 — Installing other dependencies")
    print("="*60)
    run("pip install pandas tqdm huggingface_hub numpy scikit-learn")


def download_model():
    print("\n" + "="*60)
    print("STEP 3 — Downloading GGUF model")
    print("="*60)

    LOCAL_MODEL.parent.mkdir(exist_ok=True)

    if LOCAL_MODEL.exists():
        size_mb = LOCAL_MODEL.stat().st_size / (1024 * 1024)
        print(f"[OK] Model already downloaded: {LOCAL_MODEL} ({size_mb:.0f} MB)")
        return

    print(f"[INFO] Downloading {MODEL_FILE} from {MODEL_REPO}")
    print(f"[INFO] File size: ~2.0 GB — this will take a few minutes...")

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir="models",
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Downloaded to: {path}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print()
        print("Manual download option:")
        print(f"  1. Go to: https://huggingface.co/{MODEL_REPO}")
        print(f"  2. Download: {MODEL_FILE}")
        print(f"  3. Place it at: {LOCAL_MODEL.resolve()}")
        sys.exit(1)


def verify_model():
    print("\n" + "="*60)
    print("STEP 4 — Verifying model loads and generates correctly")
    print("="*60)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("[ERROR] llama-cpp-python not importable.")
        sys.exit(1)

    if not LOCAL_MODEL.exists():
        print(f"[ERROR] Model file not found: {LOCAL_MODEL}")
        sys.exit(1)

    print(f"[INFO] Loading {LOCAL_MODEL} with n_gpu_layers=35 (full GPU offload)...")
    try:
        llm = Llama(
            model_path=str(LOCAL_MODEL),
            n_gpu_layers=35,      # Offload all 28 transformer layers + embeddings to GPU
            n_ctx=512,            # Small context for the test
            n_threads=4,
            verbose=False,
        )
        print("[OK] Model loaded successfully.")
    except Exception as e:
        print(f"[WARN] GPU load failed ({e}). Trying CPU-only...")
        llm = Llama(
            model_path=str(LOCAL_MODEL),
            n_gpu_layers=0,
            n_ctx=512,
            verbose=False,
        )
        print("[OK] Model loaded on CPU. (Will be slower but functional.)")

    # Quick generation test
    print("[INFO] Running test generation...")
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Answer with one letter only: A, B, or C."},
            {"role": "user",   "content": "Which is a fruit?\nA) rock\nB) apple\nC) car\nAnswer:"},
        ],
        max_tokens=5,
        temperature=0.0,
    )
    response = output["choices"][0]["message"]["content"].strip()
    print(f"[INFO] Test response: '{response}'")

    if "B" in response.upper():
        print("[OK] Model generating correctly — answered 'B' (apple) as expected.")
    else:
        print(f"[WARN] Unexpected response '{response}' — model may need more context.")
        print("       Pipeline will still work; this is just a sanity check.")

    del llm


def create_dirs():
    print("\n" + "="*60)
    print("STEP 5 — Creating output directories")
    print("="*60)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "dpo_model").mkdir(exist_ok=True)
    print(f"[OK] Created: {OUTPUTS_DIR}/")


def print_summary():
    print("\n" + "="*60)
    print("SETUP COMPLETE — Ready to run the pipeline")
    print("="*60)
    print()
    print("Model file  :", LOCAL_MODEL.resolve())
    print("Output dir  :", OUTPUTS_DIR.resolve())
    print()
    print("Run the pipeline in order:")
    print()
    print("  Step 1 (Baseline evaluation):")
    print("    uv run 01_evaluate_bias.py")
    print()
    print("  Step 2 (DPO fine-tuning)   — requires HuggingFace model + GPU:")
    print("    uv run 02_dpo_mitigation.py")
    print()
    print("  Step 3 (Reward calibration):")
    print("    uv run 03_reward_calibration.py")
    print()
    print("  Step 4 (Causal prompting)  — no training needed, fastest:")
    print("    uv run 04_causal_prompting.py")
    print()
    print("  Step 5 (Compare all methods):")
    print("    uv run 05_post_mitigation_eval.py")
    print()
    print("NOTE: Steps 1, 3, 4, 5 use the GGUF model (Windows-native, fast).")
    print("      Step 2 (DPO) uses HuggingFace transformers + LoRA.")
    print("      If Step 2 fails on Windows, see README for WSL2 instructions.")
    print("="*60)


if __name__ == "__main__":
    install_llama_cpp()
    install_other_deps()
    download_model()
    verify_model()
    create_dirs()
    print_summary()