#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
export OPENAI_API_BASE="http://openai.com/v1"
export OPENAI_API_KEY="sk-this-is-a-fake-key"

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "Usage: bash sh/run_all_benchmarks_vllm_openai.sh <model_name_in_model2path.json> [benchmarks_csv]"
  echo "benchmarks_csv: gpqa,hle,lcb,longbench,omni,airbench or all (default)"
  exit 1
fi

BENCHMARKS_RAW="${2:-${BENCHMARKS:-all}}"
BENCHMARKS_RAW="$(echo "$BENCHMARKS_RAW" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"

# Unified output root: results/<model>/<bench>/...
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
MODEL_SAFE="$(echo "$MODEL" | sed 's#[^A-Za-z0-9._-]#_#g')"
MODEL_RESULTS_DIR="$RESULTS_ROOT/$MODEL_SAFE"

# Common runtime knobs (can be overridden via env)
GPQA_MAX_EXAMPLES="${GPQA_MAX_EXAMPLES:-}"
GPQA_N_PROC="${GPQA_N_PROC:-16}"
GPQA_MAX_TOKENS="${GPQA_MAX_TOKENS:-32768}"

HLE_DATASET="${HLE_DATASET:-$ROOT/hle/hle_eval/data/0001.parquet}"
HLE_NUM_WORKERS="${HLE_NUM_WORKERS:-16}"
HLE_MAX_COMPLETION_TOKENS="${HLE_MAX_COMPLETION_TOKENS:-32768}"
HLE_TEMPERATURE="${HLE_TEMPERATURE:-0.0}"
HLE_MAX_SAMPLES="${HLE_MAX_SAMPLES:-}"
HLE_JUDGE_MODEL="${HLE_JUDGE_MODEL:-gpt-5.4}"
HLE_JUDGE_NUM_WORKERS="${HLE_JUDGE_NUM_WORKERS:-16}"
HLE_JUDGE_API_BASE="${HLE_JUDGE_API_BASE:-$OPENAI_API_BASE}"
HLE_JUDGE_API_KEY="${HLE_JUDGE_API_KEY:-$OPENAI_API_KEY}"
HLE_JUDGE_USE_AZURE="${HLE_JUDGE_USE_AZURE:-0}"

LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION:-release_v6}"
LCB_SCENARIO="${LCB_SCENARIO:-codegeneration}"
LCB_N="${LCB_N:-1}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-0.0}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-32768}"
LCB_NUM_PROCESS_EVALUATE="${LCB_NUM_PROCESS_EVALUATE:-16}"
LCB_TIMEOUT="${LCB_TIMEOUT:-600}"
LCB_MULTIPROCESS="${LCB_MULTIPROCESS:-16}"  # -1 means do not pass --multiprocess

LONG_N_PROC="${LONG_N_PROC:-16}"
LONG_TEMPERATURE="${LONG_TEMPERATURE:-0.0}"
LONG_COT_FLAG="${LONG_COT_FLAG:-1}"  # 1=use --cot, 0=disable
LONG_MAX_NEW_TOKENS_COT="${LONG_MAX_NEW_TOKENS_COT:-32768}"
LONG_MAX_NEW_TOKENS_DEFAULT="${LONG_MAX_NEW_TOKENS_DEFAULT:-16384}"

OMNI_DATA_FILE="${OMNI_DATA_FILE:-$ROOT/omni-math-rule/evaluation/data/omni_math_rule/test.jsonl}"
OMNI_EXP_NAME="${OMNI_EXP_NAME:-$MODEL_SAFE}"
OMNI_N_SAMPLES="${OMNI_N_SAMPLES:-1}"
OMNI_N_PROC="${OMNI_N_PROC:-16}"
OMNI_TEMPERATURE="${OMNI_TEMPERATURE:-0.0}"
OMNI_MAX_TOKENS="${OMNI_MAX_TOKENS:-32768}"

AIR_REGION="${AIR_REGION:-default}"
AIR_SAMPLE_SIZE="${AIR_SAMPLE_SIZE:-5}"
AIR_N_PROC="${AIR_N_PROC:-16}"
AIR_MAX_TOKENS="${AIR_MAX_TOKENS:-512}"
AIR_JUDGE_MODEL="${AIR_JUDGE_MODEL:-gpt-5.4}"
AIR_JUDGE_API_BASE="${AIR_JUDGE_API_BASE:-$OPENAI_API_BASE}"
AIR_JUDGE_API_KEY="${AIR_JUDGE_API_KEY:-$OPENAI_API_KEY}"
AIR_JUDGE_USE_AZURE="${AIR_JUDGE_USE_AZURE:-0}"

export OPENAI_BASE_URL="$VLLM_BASE_URL"
export OPENAI_KEY="$VLLM_API_KEY"
export HF_ENDPOINT=https://hf-mirror.com

SELECT_ALL=0
if [[ "$BENCHMARKS_RAW" == "all" ]]; then
  SELECT_ALL=1
fi

contains_bench() {
  local target="$1"
  if [[ "$SELECT_ALL" == "1" ]]; then
    return 0
  fi
  local csv=",${BENCHMARKS_RAW},"
  [[ "$csv" == *",${target},"* ]]
}

write_summary() {
  local bench="$1"
  local content="$2"
  local out_dir="$MODEL_RESULTS_DIR/$bench"
  mkdir -p "$out_dir"
  local summary_file="$out_dir/eval_summary.txt"
  printf "%b\n" "$content" > "$summary_file"
  echo "[SUMMARY] $summary_file"
}

run_gpqa() {
  echo "===== [1/6] GPQA-Diamond ====="
  cd "$ROOT/gpqa/baselines"

  local out_dir="$MODEL_RESULTS_DIR/gpqa"
  mkdir -p "$out_dir"

  CMD=(python run_baseline_diamond_vllm.py main --model_name "$MODEL" --n_proc "$GPQA_N_PROC" --max_tokens "$GPQA_MAX_TOKENS" --output_dir "$out_dir")
  if [[ -n "$GPQA_MAX_EXAMPLES" ]]; then
    CMD+=(--max_examples "$GPQA_MAX_EXAMPLES")
  fi

  "${CMD[@]}"
}

run_hle() {
  echo "===== [2/6] HLE ====="
  cd "$ROOT/hle/hle_eval"

  local out_dir="$MODEL_RESULTS_DIR/hle"
  mkdir -p "$out_dir"

  CMD=(python run_hle_pipeline.py
    --dataset "$HLE_DATASET"
    --model "$MODEL"
    --gen_api_base "$VLLM_BASE_URL"
    --gen_api_key "$VLLM_API_KEY"
    --max_completion_tokens "$HLE_MAX_COMPLETION_TOKENS"
    --temperature "$HLE_TEMPERATURE"
    --gen_num_workers "$HLE_NUM_WORKERS"
    --judge_model "$HLE_JUDGE_MODEL"
    --judge_api_base "$HLE_JUDGE_API_BASE"
    --judge_num_workers "$HLE_JUDGE_NUM_WORKERS"
    --output_dir "$out_dir")

  if [[ -n "$HLE_MAX_SAMPLES" ]]; then
    CMD+=(--max_samples "$HLE_MAX_SAMPLES")
  fi
  if [[ -n "$HLE_JUDGE_API_KEY" ]]; then
    CMD+=(--judge_api_key "$HLE_JUDGE_API_KEY")
  fi
  if [[ "$HLE_JUDGE_USE_AZURE" == "1" ]]; then
    CMD+=(--use_azure)
  fi

  "${CMD[@]}"
}

run_lcb() {
  echo "===== [3/6] LiveCodeBench v6 ====="
  cd "$ROOT/LiveCodeBench"

  local out_root="$MODEL_RESULTS_DIR/lcb"
  mkdir -p "$out_root"

  CMD=(python -m lcb_runner.runner.main
    --model "$MODEL"
    --scenario "$LCB_SCENARIO"
    --release_version "$LCB_RELEASE_VERSION"
    --n "$LCB_N"
    --temperature "$LCB_TEMPERATURE"
    --max_tokens "$LCB_MAX_TOKENS"
    --evaluate
    --num_process_evaluate "$LCB_NUM_PROCESS_EVALUATE"
    --timeout "$LCB_TIMEOUT"
    --output_root "$out_root")

  if [[ "$LCB_MULTIPROCESS" != "-1" ]]; then
    CMD+=(--multiprocess "$LCB_MULTIPROCESS")
  fi

  "${CMD[@]}"

  local latest_eval
  latest_eval="$(ls -1t "$out_root"/output/*/*_eval.json 2>/dev/null | head -n 1 || true)"
  local summary_file="$out_root/eval_summary.txt"
  if [[ -n "$latest_eval" && -f "$latest_eval" ]]; then
    {
      echo "LiveCodeBench summary"
      echo "model: $MODEL"
      echo "scenario: $LCB_SCENARIO"
      echo "eval_file: $latest_eval"
      echo
      echo "metrics_json:"
      cat "$latest_eval"
    } > "$summary_file"
  else
    {
      echo "LiveCodeBench summary"
      echo "model: $MODEL"
      echo "scenario: $LCB_SCENARIO"
      echo "eval_file: <not found>"
    } > "$summary_file"
  fi
  echo "Summary file: $summary_file"
}

run_longbench() {
  echo "===== [4/6] LongBench ====="
  cd "$ROOT/LongBench"

  local out_dir="$MODEL_RESULTS_DIR/longbench"
  mkdir -p "$out_dir"

  CMD=(python pred_openai.py --model "$MODEL" --save_dir "$out_dir" --n_proc "$LONG_N_PROC" --temperature "$LONG_TEMPERATURE" --max_new_tokens_cot "$LONG_MAX_NEW_TOKENS_COT" --max_new_tokens_default "$LONG_MAX_NEW_TOKENS_DEFAULT")
  if [[ "$LONG_COT_FLAG" == "1" ]]; then
    CMD+=(--cot)
  fi

  "${CMD[@]}"
}

run_omni_math_rule() {
  echo "===== [5/6] Omni-Math-Rule ====="
  cd "$ROOT/omni-math-rule"

  local out_root="$MODEL_RESULTS_DIR/omni"
  mkdir -p "$out_root"

  python run_test_pipeline.py \
    --model "$MODEL" \
    --data_file "$OMNI_DATA_FILE" \
    --exp_name "$OMNI_EXP_NAME" \
    --n_samples "$OMNI_N_SAMPLES" \
    --n_proc "$OMNI_N_PROC" \
    --temperature "$OMNI_TEMPERATURE" \
    --max_tokens "$OMNI_MAX_TOKENS" \
    --output_root "$out_root"
}

run_airbench() {
  echo "===== [6/6] AirBench ====="
  cd "$ROOT/air-bench-2024"

  local out_dir="$MODEL_RESULTS_DIR/airbench"
  mkdir -p "$out_dir"

  CMD=(python run_vllm.py
    --model "$MODEL"
    --region "$AIR_REGION"
    --sample_size "$AIR_SAMPLE_SIZE"
    --n_proc "$AIR_N_PROC"
    --max_tokens "$AIR_MAX_TOKENS"
    --output_dir "$out_dir"
    --vllm_url "$VLLM_BASE_URL"
    --vllm_api_key "$VLLM_API_KEY"
    --judge_api_base "$AIR_JUDGE_API_BASE"
    --judge_model "$AIR_JUDGE_MODEL")

  if [[ -n "$AIR_JUDGE_API_KEY" ]]; then
    CMD+=(--judge_api_key "$AIR_JUDGE_API_KEY")
  fi
  if [[ "$AIR_JUDGE_USE_AZURE" == "1" ]]; then
    CMD+=(--use_azure)
  fi

  "${CMD[@]}"
}

if contains_bench "gpqa"; then
  run_gpqa
fi

if contains_bench "hle"; then
  run_hle
fi

if contains_bench "lcb"; then
  run_lcb
fi

if contains_bench "longbench"; then
  run_longbench
fi

if contains_bench "omni"; then
  run_omni_math_rule
fi

if contains_bench "airbench"; then
  run_airbench
fi

echo "===== Done: selected benchmark pipelines finished for model: $MODEL ====="
echo "===== Results root: $MODEL_RESULTS_DIR ====="
