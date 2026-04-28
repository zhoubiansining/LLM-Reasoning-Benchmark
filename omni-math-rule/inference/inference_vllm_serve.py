from tqdm import tqdm
from datasets import load_from_disk
import time
import math
import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import transformers
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MAX_INT = sys.maxsize
import pdb
from openai import OpenAI

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
client = OpenAI(
    base_url=URL,
    api_key=API_KEY,
    timeout=4800,
)

# 线程锁用于保护文件写入
file_lock = threading.Lock()


def generate_dataset(dataset_path, messages_template):
    query = []
    with open(dataset_path) as f:
        for line in f.readlines():
            query.append(json.loads(line))

    messages = []
    for line in query:
        message_line = messages_template.copy()
        message_line[1]['content'] = line['problem']
        messages.append(message_line)

    return messages


def call_client_with_retries(
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        stop: str = None,
        max_tokens: int = 8192,
        top_p: float = 1.0,
) -> str:
    num_retries = 0
    content = None
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
            if content is None:
                if hasattr(completion.choices[0].message, 'reasoning_content'):
                    content = completion.choices[0].message.reasoning_content
        except Exception as e:
            if num_retries == 3:
                raise e
            print(
                f"Error calling model {model}: {e}, sleeping for {math.pow(3, num_retries)} seconds and retrying...")
            time.sleep(math.pow(3, num_retries))
            num_retries += 1
            continue
        break

    if content is None:
        content = ""
    return content


def process_item(model, messages_template, prompt_answer, ques_item, idx, max_tokens, temperature, n_samples):
    """
    处理单个数据项的函数，用于多线程调用
    """
    try:
        completions = []
        for _ in range(n_samples):
            completion = call_client_with_retries(
                model=model,
                messages=messages_template,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0
            )
            completions.append(completion)
        return idx, ques_item, completions, prompt_answer
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        return idx, ques_item, [""] * n_samples, prompt_answer


def vllm_test(args, model, data_path, start=0, end=MAX_INT, max_tokens=8192):
    gsm8k_ins = []
    gsm8k_answers = []
    gsm8k_ques_items = []

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            try:
                messages_template = [
                    {"role": "system",
                     "content": "You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \\boxed{}."},
                    {"role": "user", "content": item['question']}
                ]
            except:
                messages_template = [
                    {"role": "system",
                     "content": "You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \\boxed{}."},
                    {"role": "user", "content": item['problem']}
                ]
            gsm8k_ins.append(messages_template)
            gsm8k_ques_items.append(item)
            try:
                temp_ans = item['solution']
            except:
                temp_ans = item['answer']
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_ques_items = gsm8k_ques_items[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))

    # 使用多线程进行评测
    res_completions = [[] for _ in range(len(gsm8k_ins))]
    processed_items = [None] * len(gsm8k_ins)
    prompt_answers = [None] * len(gsm8k_ins)

    # 使用线程池执行器
    with ThreadPoolExecutor(max_workers=args.n_proc) as executor:  # 可根据需要调整线程数
        # 提交任务
        future_to_index = {
            executor.submit(
                process_item,
                model,
                messages_template,
                prompt_answer,
                ques_item,
                idx,
                max_tokens,
                args.temperature,
                args.n_samples
            ): idx
            for idx, (messages_template, prompt_answer, ques_item)
            in enumerate(zip(gsm8k_ins, gsm8k_answers, gsm8k_ques_items))
        }

        # 收集结果
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing"):
            idx, ques_item, completion, prompt_answer = future.result()
            res_completions[idx] = completion
            processed_items[idx] = ques_item
            prompt_answers[idx] = prompt_answer

    # 保存结果
    with open(args.save_path, 'w', encoding='utf-8') as f:
        for idx, (source_js, completions, prompt_answer) in enumerate(
                zip(processed_items, res_completions, prompt_answers)):
            source_js['model_generation'] = completions[0] if len(completions) > 0 else ""
            source_js['model_generations'] = completions
            source_js['num_samples'] = args.n_samples
            f.write(json.dumps(source_js, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--save_path", type=str, default='')  # saving path
    parser.add_argument("--n_proc", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vllm_test(args=args, model=args.model, data_path=args.data_file, start=args.start, end=args.end, max_tokens=args.max_tokens)
