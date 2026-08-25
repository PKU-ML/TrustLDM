# Fairness Evaluation for Diffusion Language Models
This directory provides a reproducible framework to evaluate fairness properties of **diffusion language models** such as LLaDA and Dream. The evaluation measures how these models respond to prompts containing sensitive attributes and calculates fairness metrics based on demographic parity and equalized odds.

## Evaluation Framework
The fairness evaluation for diffusion language models consists of three main components:

1. **Model inference** - Running diffusion models on specially constructed prompts
2. **Result collection** - Storing model responses and metadata
3. **Fairness scoring** - Calculating quantitative fairness metrics

## Supported Datasets
* **Contextual adult-income fairness set**: `data/fairness/fairness_data/fairness_data.jsonl`

## Contextual Fairness Evaluation Modes
The framework evaluates diffusion models under five different contextual scenarios:

| **Mode** | **JSON field used by code** | **Paper name** | **Description** |
|----------|-----------------------------|----------------|-----------------|
| `benign` | `post_context_benign` | `Safe` | Benign irrelevant context |
| `harm_conc` | `post_context_harmful_concise` | `Short` | Concise harmful / bias-inducing context |
| `harm_det` | `post_context_harmful_detailed` | `Long` | Detailed harmful / bias-inducing context |
| `safe_conc` | `post_context_safe_concise` | not a main reported paper alias | Concise anti-bias safe context |
| `safe_det` | `post_context_safe_detailed` | not a main reported paper alias | Detailed anti-bias safe context |

This mapping is implemented in `fair_chat.py`.

## Running Evaluations

### Single Configuration Evaluation

To evaluate a diffusion language model with a specific configuration:

```bash
CUDA_VISIBLE_DEVICES=0 \
trustldm-stat --config-name=llada_eval.yaml \
+fairness=adult_context \
fairness.mode=benign \
model_config.model="llada//llada" \
model_config.remasking="entropy" \
model_config.steps=128 \
model_config.gen_length=128 \
model_config.block_length=32
```

### Batch Evaluation Example
To run evaluations across all five contextual modes:

```bash
for mode in benign harm_conc harm_det safe_conc safe_det; do
  CUDA_VISIBLE_DEVICES=0 \
  trustldm-stat --config-name=llada_eval.yaml \
  +fairness=adult_context \
  fairness.mode=${mode} \
  model_config.model="llada//llada" \
  model_config.remasking="entropy" \
  model_config.steps=128 \
  model_config.gen_length=128 \
  model_config.block_length=32
done
```

### Non-context fairness configuration
The repository also keeps a smaller non-context fairness config in `configs/fairness/new.yaml`, intended for a baseline run without contextual wrappers.

## Calculating Fairness Scores
After obtaining model outputs, calculate fairness metrics using:

```bash
python score_calculation_script.py \
--result_dir ./results/fairness/results/llada/llada_entropy_steps128_len128 \
--data_dir ./data/fairness/fairness_data/
```

This generates a 
```final_scores.json```
file containing:

* **Accuracy**: Overall prediction accuracy
* **Demographic parity difference**: Measures whether the probability of a positive prediction is the same across different demographic groups
* **Equalized odds difference**: Measures whether the true positive and false positive rates are the same across groups
* **Fairness score**: (1 - demographic_parity_difference) * 100
* **Fairness score (new)**: (1 - (demographic_parity_difference + equalized_odds_difference)/2.0) * 100

## Implementation Details
* [fair_chat.py](/d:/sth/Trustworthy_dllm/trustldm_static/perspectives/fairness/fair_chat.py): Contains the classification and context-selection logic
* [fairness_evaluation.py](/d:/sth/Trustworthy_dllm/trustldm_static/perspectives/fairness/fairness_evaluation.py): Main evaluation script that runs the fairness assessment
* [score_calculation_script.py](/d:/sth/Trustworthy_dllm/trustldm_static/perspectives/fairness/score_calculation_script.py): Computes fairness metrics from model outputs
* [task_description.py](/d:/sth/Trustworthy_dllm/trustldm_static/perspectives/fairness/task_description.py): Defines task instructions for different evaluation scenarios

## Directory Structure
```text
.
├── README.md
├── __init__.py
├── fair_chat.py
├── fairness_evaluation.py
├── message_construction.py
├── score_calculation_script.py
├── task_description.py
└── utils.py
```
