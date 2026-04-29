# LLM Reasoning 统一评测 Pipeline 使用说明

本仓库提供一个统一脚本，一次性跑同一模型在多个 benchmark 上的评测（基于 vLLM serve + OpenAI Compatible API）。

当前已接入：

- GPQA-Diamond
- HLE
- LiveCodeBench
- LongBench
- Omni-Math-Rule
- AirBench

---

## 1. 你实际需要改的地方（最重要）

日常评测时，通常只需要修改这三类配置：

1) **全局模型配置（2 个 config）**

- `config/model2path.json`：模型名 -> 实际服务模型路径/ID 映射
- `config/model2maxlen.json`：模型名 -> 最大上下文长度映射

2) **统一脚本中的实验设置**

- `sh/run_all_benchmarks_vllm_openai.sh` 里的默认参数（并发、温度、token 上限、采样数等）

3) **API 信息**

- vLLM 推理 API（`VLLM_BASE_URL`, `VLLM_API_KEY`）
- HLE / AirBench 的 Judge API（`*_JUDGE_API_BASE`, `*_JUDGE_API_KEY`, `*_JUDGE_MODEL`）

> 结论：真正需要用户改的就是 **两个 config + bash 中实验参数和 API 信息**。

---

## 2. 一键运行方法

主入口脚本：

- `sh/run_all_benchmarks_vllm_openai.sh`

用法：

```bash
bash sh/run_all_benchmarks_vllm_openai.sh <model_name_in_model2path.json> [benchmarks_csv]
```

示例：

```bash
# 跑全部 benchmark
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B all

# 只跑部分 benchmark
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B gpqa,hle,lcb
```

可选 benchmark 名称：

- `gpqa`
- `hle`
- `lcb`
- `longbench`
- `omni`
- `airbench`
- `all`

---

## 3. 输出目录结构

统一输出目录：

```text
results/<model>/
```

子目录按 benchmark 分开：

```text
results/<model>/gpqa/
results/<model>/hle/
results/<model>/lcb/
results/<model>/longbench/
results/<model>/omni/
results/<model>/airbench/
```

每个 benchmark 都会尽量提供 `eval_summary.txt`（包含 acc / score / metrics 路径等核心信息），便于快速查看结果。

---

## 4. 推荐的最小配置流程

### 第一步：改模型映射

编辑：

- `config/model2path.json`
- `config/model2maxlen.json`

确保你要评测的模型名（例如 `Qwen3.5-9B`）在这两个文件中都有对应项。

### 第二步：启动 vLLM 服务

确保服务可通过 OpenAI-Compatible 接口访问（例如 `http://127.0.0.1:8000/v1`）。

### 第三步：设置 API 与实验参数

可以直接改 `sh/run_all_benchmarks_vllm_openai.sh` 默认值，或者临时用环境变量覆盖。

---

## 5. 常用环境变量（覆盖 bash 默认值）

### 5.1 vLLM 推理 API

- `VLLM_BASE_URL`（默认 `http://127.0.0.1:8000/v1`）
- `VLLM_API_KEY`（默认 `token-abc123`）

### 5.2 HLE Judge API

- `HLE_JUDGE_API_BASE`
- `HLE_JUDGE_API_KEY`
- `HLE_JUDGE_MODEL`
- `HLE_JUDGE_USE_AZURE`（`0/1`）

### 5.3 AirBench Judge API

- `AIR_JUDGE_API_BASE`
- `AIR_JUDGE_API_KEY`
- `AIR_JUDGE_MODEL`
- `AIR_JUDGE_USE_AZURE`（`0/1`）

### 5.4 LongBench token 配置

- `LONG_MAX_NEW_TOKENS_COT`（默认 `32768`）
- `LONG_MAX_NEW_TOKENS_DEFAULT`（默认 `16384`）

---

## 6. 示例：带 API 配置运行

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

## 7. 各 benchmark 说明（简要）

- **GPQA-Diamond**：输出答题日志、CSV、summary（accuracy/refusal）
- **HLE**：完整 pipeline（生成 + Judge），输出 judged 文件与 summary
- **LCB v6**：完成生成与评测，脚本自动抽取 eval json 生成 summary
- **LongBench**：输出 jsonl 结果并统计 accuracy summary
- **Omni-Math-Rule**：完整推理+评估流程，汇总 metrics 到 summary
- **AirBench**：推理+Judge 评测，输出 result 与平均分 summary

---

## 8. 常见问题

### Q1: 报模型找不到

检查 `config/model2path.json` 是否包含你传入的模型名 key。

### Q2: Judge 阶段失败

检查：

- `*_JUDGE_API_BASE`
- `*_JUDGE_API_KEY`
- `*_JUDGE_MODEL`
- 是否需要 `*_JUDGE_USE_AZURE=1`

### Q3: 只想跑单个 benchmark

第二个参数传单项即可，例如：

```bash
bash sh/run_all_benchmarks_vllm_openai.sh Qwen3.5-9B hle
```

---
