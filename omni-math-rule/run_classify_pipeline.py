import argparse
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run test-time sampling and classify questions by per-question accuracy."
    )
    parser.add_argument("--model", type=str, required=True, help="Model name/path served by vLLM OpenAI API.")
    parser.add_argument(
        "--data_file",
        type=str,
        default="",
        help="Input jsonl test file path. Default: evaluation/data/<data_name>/test_full.jsonl",
    )
    parser.add_argument("--data_name", type=str, default="math", help="Dataset name for parser/evaluator.")

    parser.add_argument("--n_samples", type=int, default=8, help="Number of generations per question.")
    parser.add_argument("--n_proc", type=int, default=8, help="Thread workers for per-question sampling.")
    parser.add_argument("--k_bins", type=int, default=5, help="Split [0,1] into k equal intervals.")
    parser.add_argument(
        "--per_level_cap",
        type=int,
        default=200,
        help="Max number of kept questions for each difficulty level. <=0 means unlimited.",
    )
    parser.add_argument("--max_questions", type=int, default=sys.maxsize, help="Maximum number of questions to test.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16384)

    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="token-abc123")

    parser.add_argument("--output_root", type=str, default="./gendata", help="Root output directory.")
    parser.add_argument("--output_subdir", type=str, default="", help="Optional subdir under output_root.")
    parser.add_argument("--save_meta", action="store_true", help="Append classification metadata to saved samples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle all data before classification.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed used when --shuffle is enabled.")
    return parser.parse_args()


def _safe_tag(text: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    sanitized = "".join(ch if ch in allowed else "_" for ch in text)
    sanitized = sanitized.strip("_")
    return sanitized or "dataset"


def _load_eval_helpers(root_dir: str):
    eval_dir = os.path.join(root_dir, "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from parser import parse_ground_truth, run_execute  # type: ignore
    from grader import math_equal  # type: ignore
    return parse_ground_truth, run_execute, math_equal


def _get_question_text(item: Dict) -> str:
    if "question" in item:
        return item["question"]
    if "problem" in item:
        return item["problem"]
    raise KeyError("Each sample must contain `question` or `problem`.")


def _build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": question},
    ]


def call_client_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    stop: str = None,
    max_tokens: int = 8192,
    top_p: float = 1.0,
) -> str:
    num_retries = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stop=stop,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            content = completion.choices[0].message.content
            if content is None and hasattr(completion.choices[0].message, "reasoning_content"):
                content = completion.choices[0].message.reasoning_content
            return content or ""
        except Exception as e:
            if num_retries >= 3:
                raise e
            wait_s = math.pow(3, num_retries)
            print(f"Error calling model {model}: {e}, sleep {wait_s}s and retry...")
            time.sleep(wait_s)
            num_retries += 1


def _sample_parallel(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    n_samples: int,
    n_proc: int,
    temperature: float,
    max_tokens: int,
) -> List[str]:
    def _one_sample() -> str:
        try:
            return call_client_with_retries(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
            )
        except Exception as e:
            print(f"[WARN] model call failed, fallback to empty string: {e}")
            return ""

    results = [None] * n_samples
    with ThreadPoolExecutor(max_workers=min(n_proc, n_samples)) as executor:
        future_to_idx = {executor.submit(_one_sample): i for i in range(n_samples)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[WARN] sample future failed, fallback to empty string: {e}")
                results[idx] = ""
    return [r or "" for r in results]


def _bucket_index(r: float, k_bins: int) -> int:
    return min(k_bins - 1, int(r * k_bins))


def _bucket_range(level_idx: int, k_bins: int) -> Tuple[float, float]:
    return level_idx / k_bins, (level_idx + 1) / k_bins


def main():
    args = parse_args()
    if args.k_bins <= 0:
        raise ValueError("--k_bins must be > 0")
    if args.n_samples <= 0:
        raise ValueError("--n_samples must be > 0")
    if args.n_proc <= 0:
        raise ValueError("--n_proc must be > 0")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    parse_ground_truth, run_execute, math_equal = _load_eval_helpers(root_dir)

    data_file = args.data_file.strip() or os.path.join(root_dir, "evaluation", "data", args.data_name, "test_full.jsonl")
    data_file = os.path.abspath(data_file)

    output_root = os.path.abspath(args.output_root)
    out_subdir = _safe_tag(args.output_subdir or args.data_name)
    out_dir = os.path.join(output_root, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=4800)

    writers = []
    level_files = []
    for level_idx in range(args.k_bins):
        low, high = _bucket_range(level_idx, args.k_bins)
        fname = f"level_{level_idx + 1}_{low:.2f}_{high:.2f}.jsonl" if level_idx < args.k_bins - 1 else f"level_{level_idx + 1}_{low:.2f}_1.00.jsonl"
        fpath = os.path.join(out_dir, fname)
        level_files.append(fpath)
        writers.append(open(fpath, "w", encoding="utf-8"))

    kept_per_level = [0] * args.k_bins
    seen_per_level = [0] * args.k_bins
    sum_r_seen_per_level = [0.0] * args.k_bins
    sum_r_kept_per_level = [0.0] * args.k_bins

    tested = 0
    skipped_before_start = 0

    print("[INFO] data_file:", data_file)
    print("[INFO] output_dir:", out_dir)
    print("[INFO] n_samples:", args.n_samples, "n_proc:", args.n_proc, "k_bins:", args.k_bins, "per_level_cap:", args.per_level_cap)

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            raw_items = [json.loads(line) for line in f if line.strip()]

        if args.shuffle:
            rng = random.Random(args.shuffle_seed)
            rng.shuffle(raw_items)
            print(f"[INFO] shuffled input order with seed={args.shuffle_seed}")

        for global_idx, item in enumerate(tqdm(raw_items, desc="Classifying")):
            if global_idx < args.start:
                skipped_before_start += 1
                continue
            if tested >= args.max_questions:
                break

            try:
                question = _get_question_text(item)
            except Exception as e:
                print(f"[WARN] skip idx={global_idx}, cannot find question/problem: {e}")
                continue

            tmp_for_gt = copy.deepcopy(item)
            if args.data_name == "omni-math" and "solution" not in tmp_for_gt and "answer" in tmp_for_gt:
                tmp_for_gt["solution"] = tmp_for_gt["answer"]
            elif "solution" not in tmp_for_gt and "answer" in tmp_for_gt:
                tmp_for_gt["solution"] = tmp_for_gt["answer"]

            try:
                _, gt_ans = parse_ground_truth(tmp_for_gt, args.data_name)
            except Exception as e:
                print(f"[WARN] skip idx={global_idx}, cannot parse gt: {e}")
                continue

            completions = _sample_parallel(
                client=client,
                model=args.model,
                messages=_build_messages(question),
                n_samples=args.n_samples,
                n_proc=args.n_proc,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            correct = 0
            preds = []
            for comp in completions:
                pred, _ = run_execute(None, comp, "cot", args.data_name, execute=False)
                preds.append(pred)
                if pred is not None and math_equal(pred, gt_ans):
                    correct += 1

            r = correct / args.n_samples
            level_idx = _bucket_index(r, args.k_bins)
            seen_per_level[level_idx] += 1
            sum_r_seen_per_level[level_idx] += r

            keep_this = args.per_level_cap <= 0 or kept_per_level[level_idx] < args.per_level_cap
            if keep_this:
                to_save = item if not args.save_meta else copy.deepcopy(item)
                if args.save_meta:
                    to_save["_cls_acc_r"] = r
                    to_save["_cls_num_correct"] = correct
                    to_save["_cls_num_samples"] = args.n_samples
                    to_save["_cls_level"] = level_idx + 1
                    to_save["_cls_bucket_k"] = args.k_bins
                    to_save["_cls_pred"] = preds
                writers[level_idx].write(json.dumps(to_save, ensure_ascii=False) + "\n")
                kept_per_level[level_idx] += 1
                sum_r_kept_per_level[level_idx] += r

            tested += 1
            print(f"[Q {tested}] idx={global_idx} r={r:.3f} -> level={level_idx + 1} ({'kept' if keep_this else 'dropped(cap)'})")
    finally:
        for w in writers:
            w.close()

    print("\nDone.")
    print(f"Skipped before start: {skipped_before_start}")
    print(f"Tested questions: {tested}")
    for i in range(args.k_bins):
        low, high = _bucket_range(i, args.k_bins)
        interval = f"[{low:.2f}, {high:.2f})" if i < args.k_bins - 1 else f"[{low:.2f}, 1.00]"
        avg_seen = (sum_r_seen_per_level[i] / seen_per_level[i]) if seen_per_level[i] else 0.0
        avg_kept = (sum_r_kept_per_level[i] / kept_per_level[i]) if kept_per_level[i] else 0.0
        print(
            f"Level {i + 1} {interval}: seen={seen_per_level[i]}, kept={kept_per_level[i]}, "
            f"avg_r_seen={avg_seen:.4f}, avg_r_kept={avg_kept:.4f}, file={level_files[i]}"
        )


if __name__ == "__main__":
    main()
