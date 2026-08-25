# Safety Evaluation for Diffusion Language Models

This directory provides a reproducible framework to evaluate safety properties of **diffusion language models** such as LLaDA and Dream. The evaluation measures how these models handle harmful prompts and calculates safety metrics.

## Evaluation Framework

The safety evaluation for diffusion language models consists of three main components:

1. **Model inference** - Running diffusion models on adversarial prompts
2. **Safety classification** - Using classifiers like Llama Guard or an OpenAI judge to determine if responses are safe or unsafe
3. **Safety scoring** - Calculating quantitative safety metrics

## Safety Evaluation Scenarios

The framework evaluates diffusion language models using two main safety benchmarks:

| **Benchmark** | **Description** |
|---------------|-----------------|
| `adv_bench` | Adversarial prompts to test safety boundaries |
| `jbb` | JailbreakBench prompts to test robustness |

## Context Types

Models are evaluated under different context scenarios. The code supports all six context keys below; the default configs in this repository use `completed`, `safe`, and `fraud`.

| **Config / dataset key** | **JSON field used by code** | **Paper name** | **Description** |
|--------------------------|-----------------------------|----------------|-----------------|
| `completed` | `post_context_completed` | `Short` | a summarization of what was assumed to be produced |
| `safe` | `post_context_safe` | `Safe` | Emphasize the harmful consequences of the generated content |
| `fraud` | `post_context_fraud` | `Long` | Wrap the content in a seemingly legitimate or research-justified framing |
| `benign` | `post_context_benign` | not a main reported paper alias | Benign irrelevant continuation |
| `concise` | `post_context_concise` | not a main reported paper alias | Short controlled-simulation framing |
| `detailed` | `post_context_detailed` | not a main reported paper alias | Detailed controlled-simulation framing |


The default setting is **Suffix+Prefix**. If you want to evaluate in a **Prefix-only** mode, comment out the suffix insertion logic in `safety_evaluation.py`.

## Running Evaluations

At the beginning, make sure either:

- you have a valid OpenAI key for the safety judge, or
- you update the fallback Llama-Guard path in `safety_evaluation.py`

### Single Configuration Evaluation

To evaluate a diffusion language model with a specific configuration:

```bash
CUDA_VISIBLE_DEVICES=1 \
trustldm-stat --config-name=llada_eval.yaml \
+safety=adv_bench.yaml \
safety.openai_api_key="your_openai_key" \
model_config.remasking="entropy" \
model_config.steps=256 \
model_config.gen_length=256 \
model_config.block_length=32
```

## Safety Metrics

The evaluation calculates safety metrics:

1. **Harmful Rate** - Percentage of responses classified as unsafe by the safety classifier
2. **Safety Score** - Inverse of harmful rate (`100 - harmful_rate`)

These metrics help quantify how well diffusion language models resist producing harmful content.

## Custom Model Configuration

For Dream diffusion model with different parameters:

```bash
CUDA_VISIBLE_DEVICES=9 \
trustldm-stat --config-name=dream_eval.yaml \
+safety=adv_bench.yaml \
model_config.remasking="maskgit_plus" \
model_config.steps=256 \
model_config.gen_length=256 \
model_config.add_role="user"
```

## Directory Structure

```text
.
├── README.md
├── __init__.py
├── safety_evaluation.py
└── utils.py
```
