# LLM Reasoning Unified Evaluation Pipeline Guide

This repository provides a unified script to evaluate the same model across multiple benchmarks in one run (based on `vLLM serve` + OpenAI-compatible API).

Currently integrated benchmarks:

- GPQA-Diamond
- HLE
- LiveCodeBench
- LongBench
- Omni-Math-Rule
- AirBench

---

## 1) What you actually need to modify (most important)

In normal usage, you usually only need to change these three parts:

1. **Global model configs (2 files)**

- `config/model2path.json`: map model alias -> actual served model path/ID
- `config/model2maxlen.json`: map model alias -> max context length

2. **Experiment settings in the unified bash script**

- `sh/run_all_benchmarks_vllm_openai.sh` defaults (parallelism, temperature, token limits, sample counts, etc.)

3. **API settings**

- vLLM inference API (`VLLM_BASE_URL`, `VLLM_API_KEY`)
- Judge API for HLE / AirBench (`*_JUDGE_API_BASE`, `*_JUDGE_API_KEY`, `*_JUDGE_MODEL`)

> In short: the only things users typically need to modify are **the two config files + bash experiment/API settings**.

---

## 2) One-command run

Main entry script:

- `sh/run_all_benchmarks_vllm_openai.sh`

Usage:

```bash
bash sh/run_all_benchmarks_vllm_openai.sh <model_name_in_model2path.json> [benchmarks_csv]
```

Examples:

```bash
# Run all benchmarks
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B all

# Run selected benchmarks only
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B gpqa,hle,lcb
```

Available benchmark keys:

- `gpqa`
- `hle`
- `lcb`
- `longbench`
- `omni`
- `airbench`
- `all`

---

## 3) Output directory layout

Unified output root:

```text
results/<model>/
```

Per-benchmark subfolders:

```text
results/<model>/gpqa/
results/<model>/hle/
results/<model>/lcb/
results/<model>/longbench/
results/<model>/omni/
results/<model>/airbench/
```

Each benchmark writes an `eval_summary.txt` whenever possible (acc/score/metrics path, etc.) for quick inspection.

---

## 4) Recommended minimal setup workflow

### Step 1: Update model mappings

Edit:

- `config/model2path.json`
- `config/model2maxlen.json`

Make sure your target model key (e.g. `Qwen3.5-9B`) exists in both files.

### Step 2: Start vLLM server

Ensure your service is reachable through an OpenAI-compatible endpoint (e.g. `http://127.0.0.1:8000/v1`).

### Step 3: Set API + experiment parameters

Either edit defaults in `sh/run_all_benchmarks_vllm_openai.sh`, or override with environment variables at runtime.

---

## 5) Common environment variables (override bash defaults)

### 5.1 vLLM inference API

- `VLLM_BASE_URL` (default: `http://127.0.0.1:8000/v1`)
- `VLLM_API_KEY` (default: `token-abc123`)

### 5.2 HLE judge API

- `HLE_JUDGE_API_BASE`
- `HLE_JUDGE_API_KEY`
- `HLE_JUDGE_MODEL`
- `HLE_JUDGE_USE_AZURE` (`0/1`)

### 5.3 AirBench judge API

- `AIR_JUDGE_API_BASE`
- `AIR_JUDGE_API_KEY`
- `AIR_JUDGE_MODEL`
- `AIR_JUDGE_USE_AZURE` (`0/1`)

### 5.4 LongBench token settings

- `LONG_MAX_NEW_TOKENS_COT` (default: `32768`)
- `LONG_MAX_NEW_TOKENS_DEFAULT` (default: `16384`)

---

## 6) Example run with API settings

```bash
VLLM_BASE_URL="http://127.0.0.1:8000/v1" \
VLLM_API_KEY="token-abc123" \
HLE_JUDGE_API_BASE="https://api.openai.com/v1/" \
HLE_JUDGE_API_KEY="sk-xxx" \
HLE_JUDGE_MODEL="o3-mini-2025-01-31" \
AIR_JUDGE_API_BASE="https://api.openai.com/v1/" \
AIR_JUDGE_API_KEY="sk-xxx" \
AIR_JUDGE_MODEL="gpt-4o" \
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B all
```

---

## 7) Benchmark notes (brief)

- **GPQA-Diamond**: writes logs, CSV, and summary (accuracy/refusal)
- **HLE**: full pipeline (generation + judge), writes judged file and summary
- **LCB v6**: runs generation + evaluation; script extracts eval JSON into summary
- **LongBench**: writes jsonl outputs and computes accuracy summary
- **Omni-Math-Rule**: full inference + evaluation; metrics summarized
- **AirBench**: inference + judge evaluation; writes result + average-score summary

---

## 8) FAQ

### Q1: "Model not found"

Check whether your model key exists in `config/model2path.json`.

### Q2: Judge stage fails

Check:

- `*_JUDGE_API_BASE`
- `*_JUDGE_API_KEY`
- `*_JUDGE_MODEL`
- whether you need `*_JUDGE_USE_AZURE=1`

### Q3: Run only one benchmark

Pass one benchmark key as the second argument, for example:

```bash
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B hle
```

---

If you want, I can also add an automatic `results/<model>/ALL_BENCH_SUMMARY.txt` that aggregates key metrics from all benchmark summaries into one table.
