---
name: hf-best-model
description: >
  Use when the user asks about finding the best, top, or recommended model for a task,
  wants to know what AI model to use, or wants to compare models by benchmark scores.
  Triggers on: "best model for X", "what model should I use for", "top models for [task]",
  "which model runs on my laptop/machine/device", "recommend a model for", "what LLM should
  I use for", "compare models for", "what's state of the art for", or any question about
  choosing an AI model for a specific use case. Always use this skill when the user wants
  model recommendations or comparisons, even if they don't explicitly mention HuggingFace
  or benchmarks.
---

# HuggingFace Best Model Finder

Finds the best models for a task by querying official HF benchmark leaderboards, enriching
results with model size data, filtering for what fits on the user's device, and returning a
comparison table with benchmark scores plus how-to-run snippets.

---

## Step 1: Parse the request

Extract from the user's message:
- **Task**: what they want the model to do (coding, math/reasoning, chat, OCR, RAG/retrieval, speech recognition, image classification, multimodal, agents, etc.)
- **Device**: hardware constraints (MacBook M-series 8/16/32/64GB unified memory, RTX GPU with VRAM amount, CPU-only, cloud/no constraint, etc.)

If device is not mentioned, skip filtering entirely and return the highest-performing models regardless of size. If the task is genuinely ambiguous, ask one clarifying question.

### Device → max parameter budget

When a device is specified, extract its available memory (unified RAM for Apple Silicon, VRAM for discrete GPUs) and apply:

- **fp16 max params (B)** ≈ memory (GB) ÷ 2
- **Q4 max params (B)** ≈ memory (GB) × 2

Examples: 16GB → 8B fp16 / 32B Q4 — 24GB VRAM → 12B fp16 / 48B Q4 — 8GB → 4B fp16 / 16B Q4

---

## Step 2: Find relevant benchmark datasets

Fetch the full list of official HF benchmarks:

```bash
curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "https://huggingface.co/api/datasets?filter=benchmark:official&limit=500" | jq '[.[] | {id, tags, description}]'
```

Read the returned list and select the datasets most relevant to the user's task — match on dataset id, tags, and description. Use your judgment; don't limit yourself to 2-3. Aim for comprehensive coverage: if 5 benchmarks clearly cover the task, use all 5.

---

## Step 3: Fetch top models from leaderboards

For each selected benchmark dataset:

```bash
curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "https://huggingface.co/api/datasets/<namespace>/<repo>/leaderboard" | jq '[.[:15] | .[] | {rank, modelId, value, verified}]'
```

Collect model IDs and scores across all benchmarks. If a leaderboard returns an error (404, 401, etc.), skip it and note it in the output.

---

## Step 4: Enrich with model metadata

For the top 10-15 candidate model IDs, use the `hub_repo_details` MCP tool:

```
hub_repo_details(repo_ids=["org/model1", "org/model2", ...], repo_type="model")
```

Extract from each response:
- **Parameters**: `safetensors.total` → convert to B (e.g., 7_241_748_480 → "7.2B")
- **License**: from model card tags (look for `license:apache-2.0`, `license:mit`, etc.)
- If `safetensors` is absent, parse size from the model name (look for "7b", "8b", "13b", "70b", "72b", etc.)

---

## Step 5: Filter and rank

**If a device was specified:**
1. Remove models exceeding the fp16 parameter budget for the device
2. Flag models that fit only with Q4 quantization (multiply budget by ~4 for Q4 capacity)
3. If a highly-ranked model is slightly over budget, keep it with a "needs Q4" note — don't silently drop it

**If no device was mentioned:** skip all size filtering — just rank by benchmark score.

Then: rank by benchmark score (descending), keep top 5-8 models.

Include proprietary models (GPT-4, Claude, Gemini) if they appear on leaderboards, but flag them as "API only / not self-hostable". If the user explicitly asked for local/open models only, exclude them.

---

## Step 6: Output

### Comparison table

```markdown
| # | Model | Params | [Benchmark 1] | [Benchmark 2] | License | On device |
|---|-------|--------|--------------|--------------|---------|-----------|
| ⭐1 | [org/name](https://huggingface.co/org/name) | 7B | 85.2% | — | Apache 2.0 | Yes (fp16) |
| 2 | [org/name](https://huggingface.co/org/name) | 13B | 83.1% | 71.5% | MIT | Q4 only |
| 3 | [org/name](https://huggingface.co/org/name) | 70B | 90.0% | 81.0% | Llama | Too large |
```

- Link model names to `https://huggingface.co/<model_id>`
- Use `—` for benchmarks where the model wasn't evaluated
- Star the top recommended pick with ⭐
- "On device" values: `Yes (fp16)`, `Q4 only`, `Too large`, `API only`

### How to run

After the table, add **## How to run** for the top 2 self-hostable picks:

````markdown
## How to run

### ⭐ org/model-7b (recommended)

**Ollama** (easiest — downloads and runs locally):
```bash
ollama run qwen2.5:7b
```

**Transformers** (Python):
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="org/model-7b", device_map="auto")
print(pipe("Your prompt here", max_new_tokens=256)[0]["generated_text"])
```

**LM Studio / llama.cpp**: search `model-7b GGUF` at https://huggingface.co/models
````

**Ollama name mappings** (common):
- `meta-llama/Llama-3.x-*B` → `llama3.2`, `llama3.3` (match version to name)
- `mistralai/Mistral-*` → `mistral`
- `Qwen/Qwen2.5-*` → `qwen2.5`
- `Qwen/Qwen3-*` → `qwen3`
- `google/gemma-3-*` → `gemma3`
- `microsoft/Phi-*` → `phi4`
- `deepseek-ai/DeepSeek-R1-*` → `deepseek-r1`
- When unsure: tell the user to check https://ollama.com/search

---

## Error handling

- **Leaderboard not found**: skip, note "leaderboard unavailable" in output
- **Model missing from hub_repo_details**: fall back to parsing size from model name
- **No benchmarks found for task**: use the curated fallback table above, or try `hub_repo_search` with `filters=["<task>"]` sorted by `trendingScore`
- **All leaderboards fail**: fall back to `hub_repo_search` for popular models tagged with the task, note that results are by popularity rather than benchmark score
