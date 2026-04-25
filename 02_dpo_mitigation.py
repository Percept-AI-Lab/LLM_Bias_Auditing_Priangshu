"""
02_dpo_mitigation.py  (Windows edition — v3)
=============================================
Step 2 — Phase A: DPO Fine-Tuning

Fixes applied vs v2:
  - Removed device_map="auto" which puts layers on meta device (breaks DPOTrainer ref copy)
  - Model now loaded directly onto cuda:0 — no meta tensors
  - warmup_ratio replaced with warmup_steps (deprecated in trl v5.2)
  - precompute_ref_log_probs=True added — avoids DPOTrainer needing a live ref model copy
  - Full version-safe DPOConfig / DPOTrainer argument inspection retained

USAGE:
    uv run 02_dpo_mitigation.py
    uv run 02_dpo_mitigation.py --epochs 1   # quick test
"""

import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="BanStereoset.csv")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", default="outputs/")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=1,
                   help="Keep at 1 on Windows to fit in 6 GB VRAM")
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--beta",       type=float, default=0.1)
    p.add_argument("--lora_r",     type=int,   default=8)
    p.add_argument("--lora_alpha", type=int,   default=16)
    p.add_argument("--max_length", type=int,   default=192)
    p.add_argument("--language",   default="both", choices=["english", "bangla", "both"])
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_deps():
    missing = []
    for pkg in ["transformers", "peft", "trl"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing: {missing}. Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    import trl, transformers, peft
    print(f"[INFO] trl={trl.__version__}  transformers={transformers.__version__}  peft={peft.__version__}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — training will be very slow on CPU.")
    else:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] GPU: {name}  ({vram:.1f} GB VRAM)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SYSTEM = "You are a helpful, unbiased assistant. Complete the sentence objectively."


def build_dpo_dataset(df: pd.DataFrame, language: str) -> Dataset:
    records = []
    pairs = []
    if language in ("english", "both"):
        pairs.append(("context", "anti_stereotype", "stereotype", "english"))
    if language in ("bangla", "both"):
        pairs.append(("bangla_context", "bangla_anti_stereotype", "bangla_stereotype", "bangla"))

    for ctx_col, anti_col, st_col, lang in pairs:
        for _, row in df.iterrows():
            if lang == "english":
                prompt = (
                    f"System: {SYSTEM}\n\nUser: Complete this sentence — replace BLANK "
                    f"with the most appropriate, unbiased word:\n{row[ctx_col]}\n\nAssistant:"
                )
            else:
                prompt = (
                    f"System: {SYSTEM}\n\nUser: বাক্যটি সম্পূর্ণ করুন — BLANK-এর জায়গায় "
                    f"নিরপেক্ষ শব্দ দিন:\n{row[ctx_col]}\n\nAssistant:"
                )
            records.append({
                "prompt":    prompt,
                "chosen":    str(row[anti_col]),
                "rejected":  str(row[st_col]),
                "bias_type": row["bias_type"],
                "language":  lang,
            })

    ds = Dataset.from_list(records)
    print(f"[INFO] DPO pairs built: {len(ds)}")
    return ds


# ---------------------------------------------------------------------------
# Model loading  — KEY FIX: no device_map="auto", load directly to cuda:0
# ---------------------------------------------------------------------------

def load_model_for_training(model_id: str, lora_r: int, lora_alpha: int):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"\n[INFO] Loading: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── CRITICAL: do NOT use device_map="auto" for DPO training ──────────
    # device_map="auto" with accelerate places some layers on a "meta" device.
    # DPOTrainer tries to copy the model to create a reference model, which
    # fails with "Cannot copy out of meta tensor; no data!".
    # Solution: load the full model onto a single device (cuda:0).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},    # Force ALL layers onto single device
        trust_remote_code=True,
    )

    # Gradient checkpointing halves activation memory at the cost of speed
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Version-safe DPO training
# ---------------------------------------------------------------------------

def train(args, dataset: Dataset, tokenizer, model, out_dir: Path):
    from trl import DPOConfig, DPOTrainer
    import trl as _trl

    model_out = out_dir / "dpo_model"
    model_out.mkdir(parents=True, exist_ok=True)

    dpo_sig     = inspect.signature(DPOConfig.__init__).parameters
    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters

    print(f"\n[INFO] trl {_trl.__version__} detected — building version-safe config...")

    # ── DPOConfig: only pass args that exist in this trl version ──────────
    cfg_kwargs = dict(
        output_dir=str(model_out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        beta=args.beta,
        remove_unused_columns=False,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        optim="adamw_torch",
        seed=args.seed,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # warmup_ratio removed in trl v5.2 — use warmup_steps instead
    if "warmup_ratio" in dpo_sig:
        cfg_kwargs["warmup_ratio"] = 0.05
    elif "warmup_steps" in dpo_sig:
        cfg_kwargs["warmup_steps"] = 10

    # max_length / max_prompt_length removed from DPOConfig in trl >= 0.9
    if "max_length" in dpo_sig:
        cfg_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in dpo_sig:
        cfg_kwargs["max_prompt_length"] = args.max_length // 2

    # precompute_ref_log_probs: compute reference log-probs in a single pass
    # before training, avoiding the need to keep a live ref model in memory.
    # This is the cleanest fix for the meta-tensor error.
    if "precompute_ref_log_probs" in dpo_sig:
        cfg_kwargs["precompute_ref_log_probs"] = True
        print("[INFO] precompute_ref_log_probs=True (avoids meta tensor error)")

    dpo_cfg = DPOConfig(**cfg_kwargs)

    # ── DPOTrainer kwargs ─────────────────────────────────────────────────
    trainer_kwargs = dict(
        model=model,
        ref_model=None,       # None = use implicit reference (log-prob precomputed)
        args=dpo_cfg,
        train_dataset=dataset,
    )

    # trl >= 0.9: 'tokenizer' renamed to 'processing_class'
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
        print("[INFO] Using processing_class= (trl >= 0.9)")
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        print("[INFO] Using tokenizer= (trl < 0.9)")

    # max_length moved into DPOTrainer in trl >= 0.9
    if "max_length" in trainer_sig and "max_length" not in cfg_kwargs:
        trainer_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in trainer_sig and "max_prompt_length" not in cfg_kwargs:
        trainer_kwargs["max_prompt_length"] = args.max_length // 2

    print("[INFO] Starting DPO training...\n")
    trainer = DPOTrainer(**trainer_kwargs)
    result  = trainer.train()

    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    print(f"\n[SAVED] LoRA adapters -> {model_out}")

    if trainer.state.log_history:
        log_df  = pd.DataFrame(trainer.state.log_history)
        log_csv = out_dir / "dpo_training_log.csv"
        log_df.to_csv(log_csv, index=False)
        print(f"[SAVED] Training log  -> {log_csv}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    check_deps()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df       = pd.read_csv(args.data)
    train_df = df.sample(frac=0.9, random_state=args.seed)
    print(f"[INFO] Training rows: {len(train_df)}")

    dataset          = build_dpo_dataset(train_df, args.language)
    tokenizer, model = load_model_for_training(args.base_model, args.lora_r, args.lora_alpha)
    result           = train(args, dataset, tokenizer, model, out_dir)

    print("\n" + "="*60)
    print("DPO TRAINING COMPLETE")
    print("="*60)
    print(f"  Final loss : {result.training_loss:.4f}")
    print(f"  Adapters   : {out_dir}/dpo_model/")
    print("\nNext: uv run 05_post_mitigation_eval.py")
    print("="*60)


if __name__ == "__main__":
    main()