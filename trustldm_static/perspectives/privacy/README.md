# Privacy Evaluation for Diffusion Language Models

This directory provides a reproducible framework to evaluate privacy properties of **diffusion language models** such as LLaDA and Dream. The evaluation measures how these models handle private information and calculates privacy leakage metrics.

## Evaluation Framework

The privacy evaluation for diffusion language models consists of three main components:

1. **Model inference** - Running diffusion models on specially constructed privacy-aware prompts
2. **Result collection** - Storing model responses and metadata
3. **Privacy scoring** - Calculating quantitative privacy leakage metrics

## Privacy Awareness Evaluation

The framework evaluates diffusion models' privacy awareness using the `awn_privacy` scenario.

| **Context Type** | **JSON field used by code** | **Paper name** | **Description** |
|------------------|-----------------------------|----------------|-----------------|
| `empty` | no extra post-context field | `Empty` | No additional context provided |
| `benign` | `post_context_benign` or `post_context_benign_user` | `Safe` | Neutral, non-sensitive context |
| `concise` | `post_context_concise` or `post_context_concise_user` | `Short` | Brief privacy-justifying context |
| `detailed` | `post_context_detailed` or `post_context_detailed_user` | `Long` | Detailed privacy-justifying context |

The code path for this mapping is `load_awareness_privacy_dataset()` in `utils.py`. When the filename contains `user`, the loader uses the `*_user` fields; otherwise it uses the non-`_user` fields.

## Running Evaluations

### Single Configuration Evaluation

To evaluate a diffusion language model with a specific configuration:

```bash
CUDA_VISIBLE_DEVICES=8 \
trustldm-stat --config-name=llada_eval.yaml \
+privacy=awn_privacy_user.yaml \
model_config.remasking="entropy" \
model_config.steps=256 \
model_config.gen_length=128 \
model_config.block_length=32 \
privacy.context_type=["benign"]
```

### Notes on privacy configs

- `awn_privacy_user.yaml` uses `data/privacy/enron_data/process_awareness_privacy_LA3_wuser.json` and evaluates `empty`, `benign`, `concise`, and `detailed`.
- `awn_privacy.yaml` uses `data/privacy/privacy_withcontext.json` and currently defaults to `concise` and `detailed`.

## Privacy Metrics

The evaluation calculates several privacy metrics:

1. **Leakage Rate** - Percentage of responses that directly reveal sensitive information
2. **Rejection Rate** - Percentage of responses that refuse to provide information (privacy-preserving)
3. **Leakage Rate without Rejection** - Leakage rate among non-rejected responses

These metrics help quantify how well diffusion language models protect private information while maintaining utility.

## Custom Model Configuration

For Dream diffusion model with different parameters:

```bash
CUDA_VISIBLE_DEVICES=8 \
trustldm-stat --config-name=dream_eval.yaml \
+privacy=awn_privacy_user.yaml \
model_config.remasking="maskgit_plus" \
model_config.steps=256 \
model_config.gen_length=128 \
model_config.add_role="user"
```

## Result Analysis

After running evaluations, results are saved in `results/privacy/generations/awareness/` and can be analyzed using:

```bash
python trustldm_static/perspectives/privacy/result_agg.py
```

This generates a comprehensive report with privacy scores for each model and context type combination.

## Directory Structure

```text
.
├── README.md
├── __init__.py
├── privacy_evaluation.py
├── result_agg.py
└── utils.py
```
