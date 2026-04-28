import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, workdir):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)



def parse_args():
    parser = argparse.ArgumentParser(description="Inference + evaluation pipeline with multi-sampling.")
    parser.add_argument("--model", type=str, required=True, help="Model name/path served by vLLM OpenAI API.")
    parser.add_argument("--data_file", type=str, required=True, help="Input jsonl test file path.")
    parser.add_argument("--exp_name", type=str, required=True, help="Base experiment name used by evaluator outputs.")
    parser.add_argument("--subset_tag", type=str, default="", help="Optional subset tag. Default: inferred from data_file name.")

    parser.add_argument("--n_samples", type=int, default=1, help="Number of generations per question.")
    parser.add_argument("--n_proc", type=int, default=8, help="Inference thread workers.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=sys.maxsize)

    parser.add_argument("--data_name", type=str, default="omni-math", help="Evaluator dataset name.")
    parser.add_argument("--prompt_type", type=str, default="cot", help="Evaluator prompt type.")
    parser.add_argument("--output_root", type=str, default="./results", help="Root dir to save inference/eval artifacts.")
    return parser.parse_args()


def _safe_tag(text):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    sanitized = "".join(ch if ch in allowed else "_" for ch in text)
    sanitized = sanitized.strip("_")
    return sanitized or "subset"


def _resolve_model_path(model_name_or_path: str) -> str:
    project_root = Path(__file__).resolve().parents[1]
    shared_config_root = project_root / "config"
    model2path_path = shared_config_root / "model2path.json"
    if not model2path_path.exists():
        return model_name_or_path

    model_map = json.loads(model2path_path.read_text(encoding="utf-8"))
    return model_map.get(model_name_or_path, model_name_or_path)


def main():
    args = parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    inference_dir = os.path.join(root_dir, "inference")
    eval_dir = os.path.join(root_dir, "evaluation")

    data_file = os.path.abspath(args.data_file)
    output_root = os.path.abspath(args.output_root)
    inference_model = _resolve_model_path(args.model)

    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    auto_subset = os.path.splitext(os.path.basename(data_file))[0]
    subset_tag = _safe_tag(args.subset_tag if args.subset_tag else auto_subset)
    data_tag = _safe_tag(args.data_name)
    run_name = f"{args.exp_name}_{data_tag}_{subset_tag}_n{args.n_samples}_{timestamp}"
    exp_name = f"{args.exp_name}__{data_tag}__{subset_tag}"

    infer_out_dir = os.path.join(output_root, "inference", data_tag, exp_name)
    os.makedirs(infer_out_dir, exist_ok=True)
    infer_out = os.path.join(infer_out_dir, f"{run_name}.jsonl")
    eval_out_dir = os.path.join(output_root, "eval", data_tag)

    inference_cmd = [
        "python",
        "inference_vllm_serve.py",
        "--model", inference_model,
        "--data_file", data_file,
        "--start", str(args.start),
        "--end", str(args.end),
        "--save_path", infer_out,
        "--n_proc", str(args.n_proc),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--n_samples", str(args.n_samples),
    ]

    eval_cmd = [
        "python",
        "math_eval.py",
        "--data_name", args.data_name,
        "--exp_name", exp_name,
        "--input_path", infer_out,
        "--output_dir", eval_out_dir,
        "--prompt_type", args.prompt_type,
    ]

    run_cmd(inference_cmd, inference_dir)
    run_cmd(eval_cmd, eval_dir)

    metrics_path = os.path.join(
        eval_out_dir,
        exp_name,
        args.data_name,
        f"math_eval_{args.prompt_type}_metrics.json",
    )

    print("\nPipeline finished.")
    print(f"Data name:        {args.data_name}")
    print(f"Data tag:         {data_tag}")
    print(f"Subset tag:       {subset_tag}")
    print(f"Effective exp:    {exp_name}")
    print(f"Inference output: {infer_out}")
    print(f"Metrics output:   {metrics_path}")

    summary_file = os.path.join(output_root, "eval_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as sf:
        sf.write("Omni-Math-Rule pipeline summary\n")
        sf.write(f"model_key: {args.model}\n")
        sf.write(f"inference_model: {inference_model}\n")
        sf.write(f"data_name: {args.data_name}\n")
        sf.write(f"subset_tag: {subset_tag}\n")
        sf.write(f"exp_name: {exp_name}\n")
        sf.write(f"inference_output: {infer_out}\n")
        sf.write(f"metrics_output: {metrics_path}\n")
        if os.path.exists(metrics_path):
            sf.write("\nmetrics_json:\n")
            with open(metrics_path, "r", encoding="utf-8") as mf:
                sf.write(mf.read())
        else:
            sf.write("\nmetrics_json: <not found>\n")

    print(f"Summary output:   {summary_file}")


if __name__ == "__main__":
    main()
