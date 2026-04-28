import os, csv, json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHARED_CONFIG_ROOT = _PROJECT_ROOT / "config"
model_map = json.loads((_SHARED_CONFIG_ROOT / 'model2path.json').read_text(encoding='utf-8'))
maxlen_map = json.loads((_SHARED_CONFIG_ROOT / 'model2maxlen.json').read_text(encoding='utf-8'))

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    max_len = maxlen_map[model] - max_new_tokens - 32
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    
    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            content = completion.choices[0].message.content
            if content is None:
                if hasattr(completion.choices[0].message, 'reasoning_content'):
                    content = completion.choices[0].message.reasoning_content
            return content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, fout):
    model = args.model
    if ("gpt" in model or "o1" in model) and "oss" not in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY,
        timeout=4800,
    )
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=args.temperature, max_new_tokens=args.max_new_tokens_cot)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=args.temperature, max_new_tokens=args.max_new_tokens_default)
        if output is None or output == '':
            print("Empty response. Skip.")
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, temperature=args.temperature, max_new_tokens=args.max_new_tokens_default)
            if output is None or output == '':
                print("Empty extraction response. Skip.")
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    total = 0
    correct = 0
    easy = hard = short = medium = long_len = 0
    easy_acc = hard_acc = short_acc = medium_acc = long_acc = 0.0

    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if "judge" not in item:
                    continue

                acc = 1.0 if item["judge"] else 0.0
                total += 1
                correct += int(acc)

                if item.get("difficulty") == "easy":
                    easy += 1
                    easy_acc += acc
                else:
                    hard += 1
                    hard_acc += acc

                length_tag = item.get("length")
                if length_tag == "short":
                    short += 1
                    short_acc += acc
                elif length_tag == "medium":
                    medium += 1
                    medium_acc += acc
                else:
                    long_len += 1
                    long_acc += acc

    def pct(numerator, denominator):
        return 100.0 * numerator / denominator if denominator > 0 else None

    overall_pct = pct(easy_acc + hard_acc, total)
    easy_pct = pct(easy_acc, easy)
    hard_pct = pct(hard_acc, hard)
    short_pct = pct(short_acc, short)
    medium_pct = pct(medium_acc, medium)
    long_pct = pct(long_acc, long_len)

    summary_path = os.path.join(args.save_dir, "eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("LongBench summary\n")
        sf.write(f"model: {args.model}\n")
        sf.write(f"output_file: {out_file}\n")
        sf.write(f"total_judged: {total}\n")
        sf.write(f"correct: {correct}\n")
        sf.write("\n")
        sf.write("Metric\tValue\n")
        sf.write(f"Overall\t{round(overall_pct, 1) if overall_pct is not None else 'N/A'}\n")
        sf.write(f"Easy\t{round(easy_pct, 1) if easy_pct is not None else 'N/A'}\n")
        sf.write(f"Hard\t{round(hard_pct, 1) if hard_pct is not None else 'N/A'}\n")
        sf.write(f"Short\t{round(short_pct, 1) if short_pct is not None else 'N/A'}\n")
        sf.write(f"Medium\t{round(medium_pct, 1) if medium_pct is not None else 'N/A'}\n")
        sf.write(f"Long\t{round(long_pct, 1) if long_pct is not None else 'N/A'}\n")

    if overall_pct is not None:
        print(f"Overall Accuracy: {overall_pct:.1f}%")
        print(
            "Breakdown - "
            f"Easy: {round(easy_pct, 1) if easy_pct is not None else 'N/A'}%, "
            f"Hard: {round(hard_pct, 1) if hard_pct is not None else 'N/A'}%, "
            f"Short: {round(short_pct, 1) if short_pct is not None else 'N/A'}%, "
            f"Medium: {round(medium_pct, 1) if medium_pct is not None else 'N/A'}%, "
            f"Long: {round(long_pct, 1) if long_pct is not None else 'N/A'}%"
        )
    else:
        print("Accuracy: N/A (no judged samples)")

    print(f"Summary file: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--temperature", "-t", type=float, default=0.0)
    parser.add_argument("--max_new_tokens_cot", type=int, default=32768,
                        help="max_new_tokens used for CoT generation")
    parser.add_argument("--max_new_tokens_default", type=int, default=16384,
                        help="max_new_tokens used for non-CoT generation and answer extraction")
    args = parser.parse_args()
    main()
    