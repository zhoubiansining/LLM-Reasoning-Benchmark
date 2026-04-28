import argparse
import os
import subprocess
import sys


def run_cmd(cmd, workdir):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full HLE pipeline: generation + judge")
    parser.add_argument("--dataset", type=str, default="data/0001.parquet", help="HLE dataset path or HF dataset name")
    parser.add_argument("--model", type=str, required=True, help="Generation model key in model2path.json")

    parser.add_argument("--gen_api_base", type=str, default="http://127.0.0.1:8000/v1", help="OpenAI-compatible API base for generation")
    parser.add_argument("--gen_api_key", type=str, default="token-abc123", help="API key for generation endpoint")
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gen_num_workers", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--judge_model", type=str, default="o3-mini-2025-01-31", help="Judge model name")
    parser.add_argument("--judge_api_base", type=str, default="https://api.openai.com/v1/", help="Judge API base URL")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Judge API key")
    parser.add_argument("--judge_num_workers", type=int, default=32)
    parser.add_argument("--use_azure", action="store_true", help="Use Azure OpenAI for judge")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generation/judge artifacts")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    workdir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(args.output_dir, exist_ok=True)
    pred_file = os.path.join(args.output_dir, f"hle_{os.path.basename(args.model)}.json")
    judged_file = os.path.join(args.output_dir, f"judged_{os.path.basename(pred_file)}")

    if not args.skip_generation:
        gen_cmd = [
            sys.executable,
            "run_model_predictions_vllm.py",
            "--dataset", args.dataset,
            "--model", args.model,
            "--max_completion_tokens", str(args.max_completion_tokens),
            "--temperature", str(args.temperature),
            "--num_workers", str(args.gen_num_workers),
            "--api_base", args.gen_api_base,
            "--api_key", args.gen_api_key,
            "--output_file", pred_file,
        ]
        if args.max_samples is not None:
            gen_cmd += ["--max_samples", str(args.max_samples)]
        run_cmd(gen_cmd, workdir)

    if not args.skip_judge:
        judge_cmd = [
            sys.executable,
            "run_judge_results.py",
            "--dataset", args.dataset,
            "--predictions", pred_file,
            "--num_workers", str(args.judge_num_workers),
            "--judge", args.judge_model,
            "--api_base", args.judge_api_base,
            "--output_file", judged_file,
        ]
        if args.judge_api_key:
            judge_cmd += ["--api_key", args.judge_api_key]
        if args.use_azure:
            judge_cmd += ["--use_azure"]
        run_cmd(judge_cmd, workdir)

    print("\nHLE pipeline finished.")
    print(f"Predictions: {pred_file}")
    print(f"Judged file: {judged_file}")


if __name__ == "__main__":
    main()
