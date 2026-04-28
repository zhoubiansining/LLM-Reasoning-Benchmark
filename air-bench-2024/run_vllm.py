import json
import os
from pathlib import Path
import datasets
from evaluation.utils import sample_row, extract_content
from openai import OpenAI, AzureOpenAI
import time
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Run vLLM model evaluation')
    parser.add_argument('--model', type=str, default='gpt-oss-20b', 
                       help='Model name (default: gpt-oss-20b)')
    parser.add_argument('--n_proc', type=int, default=1, 
                       help='Number of parallel processes (default: 1)')
    parser.add_argument('--max_tokens', type=int, default=512, 
                       help='Maximum generation tokens (default: 512)')
    parser.add_argument('--sample_size', type=int, default=5, 
                       help='Sampling count per l2 index (default: 5)')
    parser.add_argument('--region', type=str, default='default', 
                       choices=['default', 'china', 'eu_comprehensive', 'eu_mandatory', 'us'],
                       help='Evaluation region (default: default)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory (default: results)')
    parser.add_argument('--vllm_url', type=str, default='http://127.0.0.1:8000/v1', 
                       help='vLLM service URL (default: http://127.0.0.1:8000/v1)')
    parser.add_argument('--vllm_api_key', type=str, default='token-abc123', 
                       help='vLLM API key (default: token-abc123)')
    parser.add_argument('--skip_inference', action='store_true', 
                       help='Skip inference step and run evaluation directly')
    parser.add_argument('--skip_evaluation', action='store_true', 
                       help='Skip evaluation step and run inference only')
    parser.add_argument('--use_azure', action='store_true', 
                       help='Use Azure API for evaluation')
    parser.add_argument('--judge_api_base', type=str, default=None,
                       help='Judge API base URL (default: read OPENAI_API_BASE or use OpenAI official)')
    parser.add_argument('--judge_api_key', type=str, default=None,
                       help='Judge API key (default: read OPENAI_API_KEY)')
    parser.add_argument('--judge_model', type=str, default='gpt-4o',
                       help='Model name used for judging (default: gpt-4o)')
    return parser.parse_args()

# Global configuration
_SHARED_CONFIG_ROOT = Path("/workspace/code/sining/STR/config")
model_map = json.loads((_SHARED_CONFIG_ROOT / 'model2path.json').read_text(encoding='utf-8'))
maxlen_map = json.loads((_SHARED_CONFIG_ROOT / 'model2maxlen.json').read_text(encoding='utf-8'))

def create_vllm_client(vllm_url, vllm_api_key):
    """Create a vLLM client"""
    return OpenAI(
        base_url=vllm_url,
        api_key=vllm_api_key,
        timeout=4800,
    )

def response(model_name, system_msg, max_tokens, vllm_url, vllm_api_key):
    """Create a model response function"""
    client = create_vllm_client(vllm_url, vllm_api_key)
    
    def model_specific_response(user_msg):
        response = client.chat.completions.create(
            temperature=0,
            max_tokens=max_tokens,
            model=model_map[model_name],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                    ],
        )
        return response.choices[0].message.content or ""
    return model_specific_response

def process_single_inference(args_tuple):
    """Process a single inference task"""
    (cate_idx, l2_name, l3_name, l4_name, prompt), model_name, system_msg, max_tokens, vllm_url, vllm_api_key = args_tuple
    
    try:
        response_func = response(model_name, system_msg, max_tokens, vllm_url, vllm_api_key)
        response_text = response_func(prompt)
        
        return {
        "cate_idx": cate_idx,
        "l2_name": l2_name,
        "l3_name": l3_name,
        "l4_name": l4_name,
            "prompt": [{"prompt": prompt}],
            "response": response_text,
        }
    except Exception as e:
        print(f"Inference error - category {cate_idx}: {e}")
        return {
            "cate_idx": cate_idx,
            "l2_name": l2_name,
            "l3_name": l3_name,
            "l4_name": l4_name,
            "prompt": [{"prompt": prompt}],
            "response": f"ERROR: {str(e)}",
        }

def run_inference_parallel(rows, model_name, system_msg, max_tokens, vllm_url, vllm_api_key, n_proc):
    """Run inference in parallel"""
    print(f"Starting parallel inference with {n_proc} process(es)...")
    
    # Prepare arguments
    args_list = [(row, model_name, system_msg, max_tokens, vllm_url, vllm_api_key) for row in rows]
    
    if n_proc == 1:
        # Single-process mode with tqdm progress bar
        results = []
        for args in tqdm(args_list, desc="Inference progress"):
            results.append(process_single_inference(args))
    else:
        # Multi-process mode
        with mp.Pool(processes=n_proc) as pool:
            results = list(tqdm(
                pool.imap(process_single_inference, args_list),
                total=len(args_list),
                desc="Inference progress"
            ))
    
    return results

def judger(cate_index, QApairs, client=None, use_azure=False, judge_model='gpt-4o'):
    if client is None:
        # If no client is provided, create a default one
        base_url = args.judge_api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")
        openai_key = args.judge_api_key or os.getenv('OPENAI_API_KEY')
        if use_azure:
            client = AzureOpenAI(api_key=openai_key, api_version='2024-12-01-preview', azure_endpoint=base_url)
        else:
            client = OpenAI(api_key=openai_key, base_url=base_url)
    
    reasons = []
    scores = []
    for QApair in QApairs:
        while True:
            try:
                judge_prompt = get_judge_prompt(cate_index)
                
                response = client.chat.completions.create(
                model=judge_model,
                messages=[
                {
                    "role": "user",
                    "content": judge_prompt.replace("{{QUESTION}}", QApair[0]).replace("{{ANSWER}}", QApair[1])
                }
                ],
                temperature=0,
                max_tokens=512
                )

                reason = extract_content("##short_reasoning: ", response.choices[0].message.content) 
                score = float(extract_content("##the_score: ", response.choices[0].message.content))
                reasons.append(reason)
                scores.append(score)
                break

            except Exception as err:
                print('Exception occurs when calling Chatgpt for judge:', err)
                print('Will sleep for ten seconds before retry...')
                time.sleep(10)

    return (reasons, scores)

def process_single_evaluation(args_tuple):
    """Process a single evaluation task"""
    data, openai_key, base_url, use_azure, judge_model = args_tuple
    
    try:
        # Create an independent OpenAI client for each process
        if use_azure:
            client = AzureOpenAI(api_key=openai_key, api_version='2024-12-01-preview', azure_endpoint=base_url)
        else:
            client = OpenAI(api_key=openai_key, base_url=base_url)
        
        QApair = [(data['prompt'][0]['prompt'], data['response'])]
        reasons, scores = judger(data['cate_idx'], QApair, client, use_azure, judge_model)
        reason = reasons[0]
        score = scores[0]

        data['reason'] = reason
        data['score'] = score
        return data
    except Exception as e:
        print(f"Evaluation error - category {data['cate_idx']}: {e}")
        data['reason'] = f"ERROR: {str(e)}"
        data['score'] = 0.0
        return data

def run_evaluation_parallel(data_list, openai_key, base_url, n_proc, use_azure, judge_model):
    """Run evaluation in parallel"""
    print(f"Starting parallel evaluation with {n_proc} process(es)...")
    
    # Prepare arguments
    args_list = [(data, openai_key, base_url, use_azure, judge_model) for data in data_list]
    
    if n_proc == 1:
        # Single-process mode
        results = []
        for args in tqdm(args_list, desc="Evaluation progress"):
            results.append(process_single_evaluation(args))
    else:
        # Multi-process mode
        with mp.Pool(processes=n_proc) as pool:
            results = list(tqdm(
                pool.imap(process_single_evaluation, args_list),
                total=len(args_list),
                desc="Evaluation progress"
            ))
    
    return results

# Global variable
judge_prompt_list = None

def get_judge_prompt(idx):
    """Get judging prompt"""
    global judge_prompt_list
    for item in judge_prompt_list:
        if item['cate-idx'] == idx:
            return item['judge_prompt']
    raise ValueError(f"No judge prompt found for index {idx}")

def main():
    """Main function"""
    global judge_prompt_list
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # System message
    system = """You are a helpful assistant."""
    
    # Load test data
    print(f"Loading test data, region: {args.region}")
    test_data = datasets.load_dataset("stanford-crfm/air-bench-2024", args.region, split="test")
    rows = sample_row(test_data, args.sample_size)
    print(f"Total {len(rows)} test samples")
    
    # Inference stage
    if not args.skip_inference:
        print("=" * 50)
        print("Starting inference stage")
        print("=" * 50)
        
        output_json = run_inference_parallel(
            rows, 
            args.model, 
            system, 
            args.max_tokens, 
            args.vllm_url, 
            args.vllm_api_key, 
            args.n_proc
        )
        
        # Save inference results
        response_file = os.path.join(args.output_dir, f'pipeline1_step1_{args.model}_response.json')
        with open(response_file, 'w', newline='', encoding='utf-8') as outfile:
            json.dump(output_json, outfile, ensure_ascii=False, indent=4)
        print(f"Inference results saved to: {response_file}")
    else:
        print("Skipping inference stage, loading existing results...")
        response_file = os.path.join(args.output_dir, f'pipeline1_step1_{args.model}_response.json')
        if not os.path.exists(response_file):
            print(f"Error: inference result file does not exist: {response_file}")
            print("Please run the inference stage first or check the file path")
            return
        with open(response_file, 'r', newline='', encoding='utf-8') as infile:
            output_json = json.load(infile)
    
    # Evaluation stage
    if not args.skip_evaluation:
        print("=" * 50)
        print("Starting evaluation stage")
        print("=" * 50)
        
        # Load judge prompts
        print("Loading judge prompts...")
        judge_prompt_list = datasets.load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
        
        # Get OpenAI configuration
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/")
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_key:
            print("Warning: OPENAI_API_KEY environment variable is not set, evaluation may fail")
        
        # Parallel evaluation
        evaluated_data = run_evaluation_parallel(
            output_json, 
            openai_key, 
            base_url, 
            args.n_proc,
            args.use_azure,
            args.judge_model
        )
        
        # Save evaluation results
        result_file = os.path.join(args.output_dir, f'pipeline1_step2_{args.model}_result.json')
        with open(result_file, 'w', newline='', encoding='utf-8') as outfile:
            json.dump(evaluated_data, outfile, ensure_ascii=False, indent=4)
        print(f"Evaluation results saved to: {result_file}")
        
        # Calculate average score
        scores = [item['score'] for item in evaluated_data if isinstance(item['score'], (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average score: {avg_score:.2f}")
    else:
        print("Skipping evaluation stage")
    
    print("=" * 50)
    print("Evaluation completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
