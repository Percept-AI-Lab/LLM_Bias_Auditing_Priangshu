# A Novel Adaptive Three-Level Alignment for Bias Recognition and Rectification in Large Language Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct-orange)
![Dataset](https://img.shields.io/badge/Dataset-BanStereoSet-purple)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Bengali-teal)

**Priangshu Paul**

*Cross-lingual stereotype bias evaluation and mitigation in LLMs using a unified three-level adaptive debiasing framework.*

</div>

---

## Overview

This repository contains the full implementation of the paper **"A Novel Adaptive Three-Level Alignment for Bias Recognition and Rectification in Large Language Models"** — a systematic study of stereotype bias in **Qwen2.5-3B-Instruct** across nine social bias categories in both **English and Bengali**, using the **BanStereoSet** benchmark.

The proposed framework operates at three successive levels of model computation:

| Level | Method | Intervention Point |
|-------|--------|--------------------|
| **Level I** | DPO fine-tuning with LoRA | Parametric (model weights) |
| **Level II** | Reward-guided logit calibration | Output score distribution |
| **Level III** | Causal chain-of-thought prompting | Reasoning context |

The framework is **adaptive**: no single level dominates across all bias categories. Per-category level selection based on held-out validation performance consolidates the strengths of all three levels into a unified deployment strategy.

---

## Key Results

| Method | Mean SR | vs. Baseline | Cross-Lingual Gap |
|--------|---------|--------------|-------------------|
| Baseline | 0.524 | — | 0.216 |
| Level II — Calibration | 0.457 | −8.8% | 0.174 |
| Level III — Anti-Stereo CoT | 0.402 | −9.4% | 0.087 |
| Level III — Multi-Chain | 0.413 | −9.8% | 0.146 |
| **Level III — Self-Debiasing** | **0.372** | **−14.4%** | **0.075** |

> **Random baseline SR = 0.333.** Self-debiasing closes ~80% of the gap between the biased baseline and the random baseline — without any model training.

### Notable findings

- 🔴 **Region inversion:** English SR = 0.095 (sub-random) vs Bengali SR = 0.492 — a bias *entirely invisible* to English-only evaluation
- ✅ **Caste (Bengali) under calibration:** 47.1% reduction — the single largest per-category drop in the study
- ✅ **Caste (English) under self-debiasing:** 56.4% reduction
- ✅ **DPO convergence:** rewards accuracy = 0.944, log-prob gap chosen vs rejected = 26.6 nats

---

## Dataset: BanStereoSet

BanStereoSet is a bilingual (English + Bengali) extension of StereoSet covering **1,194 instances** across **9 bias categories**:

| Category | n | Notes |
|---|---|---|
| Race | 241 | Ethnicity and nationality associations |
| Profession | 206 | Occupational stereotypes |
| Gender | 178 | Sex-based role assumptions |
| Ageism | 134 | Age-based competence stereotypes |
| Beauty | 130 | Appearance-based character judgements |
| Beauty × Profession | 126 | Intersection of appearance and occupation |
| Region | 63 | South Asian regional identity stereotypes |
| Caste | 60 | South Asian caste-based stereotypes *(novel)* |
| Religion | 56 | Faith-based behavioural stereotypes |

Each item provides a context sentence with a `BLANK` token and three candidate completions: `stereotype`, `anti-stereotype`, `unrelated`.

---

## Pipeline

```
00_setup.py                ← Download model, verify GPU, create directories
01_evaluate_bias.py        ← Baseline evaluation (both languages)
02_dpo_mitigation.py       ← Level I: DPO fine-tuning with LoRA
03_reward_calibration.py   ← Level II: Reward-guided logit calibration
04_causal_prompting.py     ← Level III: CoT prompting (3 strategies)
05_post_mitigation_eval.py ← Cross-method comparison + cross-lingual analysis
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/priangshu17/llm-bias-auditing.git
cd llm-bias-auditing

# Create environment (recommended: uv)
uv venv && source .venv/bin/activate        # Linux/macOS
uv venv && .venv\Scripts\activate           # Windows

# Install dependencies
uv pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

---

## Usage

Run the pipeline in order:

```bash
# Step 0: Setup — downloads Qwen2.5-3B-Instruct Q4_K_M GGUF (~2 GB)
python 00_setup.py

# Step 1: Baseline evaluation
python 01_evaluate_bias.py

# Step 2: DPO fine-tuning (Level I) — requires ~5.8 GB VRAM, ~13 hrs on RTX 4050
python 02_dpo_mitigation.py

# Step 3: Reward calibration (Level II)
python 03_reward_calibration.py --alpha 0.6

# Step 4: Causal prompting (Level III) — runs all 3 strategies
python 04_causal_prompting.py

# Step 5: Build comparison table and cross-lingual analysis
python 05_post_mitigation_eval.py

# Skip DPO and run only inference-time methods
python 05_post_mitigation_eval.py --skip_dpo
```

---

## Output Files

| File | Description |
|------|-------------|
| `baseline_results.csv` | Per-row model predictions at baseline |
| `stereotype_ratios_baseline.csv` | SR, LMS, UR per (bias_type, language) |
| `stereotype_summary_baseline.csv` | Overall summary statistics |
| `dpo_training_log.csv` | Loss, rewards margin, rewards accuracy per step |
| `calibration_results.csv` | Per-row calibrated predictions |
| `stereotype_ratios_calibration.csv` | SR per (bias_type, language) after calibration |
| `causal_prompt_results.csv` | Per-row CoT predictions (all 3 strategies) |
| `stereotype_ratios_causal.csv` | SR per (bias_type, language, strategy) |
| `comparison_all_methods.csv` | Unified comparison across all methods |
| `cross_lingual_analysis.csv` | \|EN − BN\| gap per (bias_type, method) |

---

## Model & Hardware

| Component | Specification |
|-----------|---------------|
| Base model | Qwen2.5-3B-Instruct |
| Inference format | GGUF Q4_K_M via llama-cpp-python |
| Inference VRAM | ~2.0 GB |
| Training format | float16 + LoRA (rank=8) via HuggingFace TRL |
| Training VRAM | ~5.8 GB (with gradient checkpointing) |
| LoRA trainable params | ~14.97M / 3,100M (0.48%) |
| DPO β | 0.1 |
| Training steps | 807 (3 epochs, 2,150 preference pairs) |
| Seed | 42 (all experiments) |

---

## Reproducibility

All stochastic operations are seeded with `42` (Python `random`, NumPy, PyTorch). Greedy decoding (`temperature=0.0`) ensures deterministic inference. Per-row option shuffle mappings are saved in all output CSVs for independent auditing.

The training script uses `inspect.signature()` to dynamically detect compatible arguments for the installed `trl` version, making it forward- and backward-compatible across trl 0.8.x–5.x without version pinning.

---

## Citation

```bibtex
@article{paul2026atlas,
  title={A Novel Adaptive Three-Level Alignment for Bias Recognition
         and Rectification in Large Language Models},
  author={Paul, Priangshu},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">
<sub>Built with Qwen2.5 · BanStereoSet · llama-cpp-python · HuggingFace TRL · PEFT</sub>
</div>
