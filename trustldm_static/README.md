# TrustLDM_static: Static Evaluation for Diffusion Language Models

## Overview

TrustLDM_static is a **static evaluation suite** designed to measure the *trustworthiness* of modern **diffusion language models (LDMs)**. It provides a transparent framework to evaluate safety, privacy, and fairness behaviors of LDMs by running fixed tests.

This repository focuses on the **static evaluation mode**. It supports multiple perspectives (safety, privacy, fairness) and multiple model families.

![trustldm_static pipeline](./pipeline.png)

> ⚠️ **Safety Reminder**: TrustLDM_static is **only intended for research and defensive evaluation** of large diffusion language models. It **must not** be used to attack real systems, harm users, or generate malicious outputs outside a controlled research setting.
> **Please use responsibly.**

---

## Target Models

TrustLDM_static supports evaluation of the following language diffusion models:

### Open-Source Models
- **LLaDA-8B-Instruct** (`llada_eval.yaml`)
- **LLaDA-1.5** (`llada_1___5.yaml`)
- **LLaDA-MoE-7B-A1B-Instruct** (`llada_moe.yaml`)
- **Dream-v0-Instruct-7B** (`dream_eval.yaml`)

### Closed-Source Models
- **Mercury** (`mercury_eval.yaml`, legacy / optional code path kept in the repository)

---

## Introduction

TrustLDM_static runs evaluation pipelines for following perspective:

- **Safety**: evaluates whether an LDM produces unsafe or harmful outputs under different contextual prompts.
- **Privacy**: measures whether an LDM inadvertently reveals or leaks private information in a database-manager-style setting.
- **Fairness**: checks whether outputs can be induced to exhibit demographic bias under contextual perturbations.

Each perspective has its own data format, scoring rules, and evaluation outputs. The detailed dataset-field-to-paper-name mapping is documented in the per-perspective README files under `perspectives/*`.

---

## Environment Setup

### 1) Create a Conda env

```bash
conda create -n trustldm-static python=3.10 -y
conda activate trustldm-static
```

For newer GPU architectures, use the unified `trustldm-d2f` environment
documented in the repository root. It supports both the LLaDA family and
Dream, so the static and automatic pipelines can share one environment.

### 2) Install PyTorch

We recommend **one of the following**:

- **Option A (stable)**: `torch==2.0.1`
- **Option B (CUDA 11.8)**: `torch==2.1.0+cu118`

Example:

```bash
python -m pip install "torch==2.0.1" torchvision torchaudio -c pytorch
```

or

```bash
python -m pip install "torch==2.1.0+cu118" torchvision torchaudio -c pytorch -c nvidia
```

### 3) Install the project

```bash
pip install -e .
```

This installs dependencies defined in `setup.cfg`.

### 4) Install `crfm_helm` wheel

```bash
pip install crfm_helm-0.2.3-py3-none-any.whl
```

> ⚠️ Installing this wheel will **downgrade** some key packages (for example `transformers` and `tokenizers`) to versions compatible with `crfm_helm`. If you cannot run evaluations after installing this wheel, you should either:
> - Upgrade back to compatible versions (`transformers==4.43.2`, `tokenizers==0.19.1`), or
> - Re-run `pip install -e .` in the same env.

> ⚠️ The legacy CUDA 11.8 setup above may still require model-specific
> dependency versions. For the maintained newer-GPU setup, follow
> `requirements-trustldm-d2f.txt` instead of downgrading `transformers` or
> `tokenizers` after installing the project.


> ⚠️ **Other model conflicts**: If you want to run evaluations for other newer models (e.g., SDAR, LLaDA-2.0), make sure to check the required dependency versions and adjust your environment accordingly. As far as we are concerned, SDAR and LLADA-2.0 do not support `torch` version as low as `2.0.1` or `2.1.0+cu118`(which we recommend as above), due to the conflict between `transformers==4.57.0` and `torch`. We provide here another base env of `torch==2.7.1+cu118`. 

### Models

Some configuration files contain **absolute model paths** that you must update before running evaluations. In particular, edit `model_config.model_path` in configs such as:

- `configs/llada_eval.yaml`
- `configs/llada_1___5.yaml`
- `configs/llada_moe.yaml`
- `configs/dream_eval.yaml`

You can find all hardcoded paths with:

```bash
grep -RIl "/data1/ybshi" trustldm_static/configs
```

Make sure the paths point to your local (or remote) model checkpoint directories before running. You can also create your own `yaml` file for any model you want to evaluate. Remember to write your variables legally and update the paths accordingly.

---

## Run Evaluations

TrustLDM_static is driven by the `trustldm-stat` CLI entry point (installed via `pip install -e .`).

### Basic evaluation example

Run a single perspective (for example safety) using a specific configuration:

```bash
trustldm-stat --config-name=llada_eval.yaml \
    +safety=adv_bench.yaml \
    model_config.remasking="entropy" \
    model_config.steps=256 \
    model_config.gen_length=256 \
    model_config.block_length=32
```

### Privacy evaluation example

```bash
trustldm-stat --config-name=dream_eval.yaml \
    +privacy=awn_privacy_user.yaml \
    model_config.remasking="maskgit_plus" \
    model_config.steps=256 \
    model_config.gen_length=128
```

### Fairness evaluation example

```bash
trustldm-stat --config-name=llada_eval.yaml \
    +fairness=adult_context \
    fairness.mode=benign \
    model_config.remasking="entropy" \
    model_config.steps=128 \
    model_config.gen_length=128 \
    model_config.block_length=32
```

### Mercury evaluation example

For the legacy closed-source Mercury path:

```bash
export MERCURY_API_KEY="your_api_key_here"

trustldm-stat --config-name=mercury_eval.yaml \
    +safety=adv_bench.yaml \
    model_config.gen_length=50
```

> ⚠️ **API Key Required**: Mercury is a closed-source model. You must set the `MERCURY_API_KEY` environment variable before running evaluations.

> The config system uses Hydra. You can inspect available configs under `configs/`.

---

## Files

### Datasets

- `data/` — labeled evaluation data used by the pipelines. Every perspective has its own subfolder with the relevant data files.

### Entry point
- `main.py` — main CLI entrypoint. It loads Hydra configs and dispatches to the selected perspectives.

### Perspectives
Each perspective has its own directory and evaluation logic:

- `perspectives/safety/` — safety evaluation logic
- `perspectives/privacy/` — privacy evaluation logic
- `perspectives/fairness/` — fairness evaluation logic

Each perspective folder includes a README describing usage, dataset fields, and paper-name mappings.

### Configuration
- `configs/` — Hydra configuration templates for running evaluations.

### Important support files
- `setup.cfg` — dependency definitions and console entry point `trustldm-stat`.

### Results
- Evaluation results are saved under `results/` with subfolders for each perspective and model. For example, safety evaluation generations are saved under `results/safety/generations/`, and privacy results are saved under `results/privacy/generations/awareness/`.

### Directory Layout

```text
.
├── setup.cfg
├── main.py
├── configs/
│   ├── llada_eval.yaml
│   ├── llada_1___5.yaml
│   ├── llada_moe.yaml
│   ├── dream_eval.yaml
│   ├── mercury_eval.yaml
│   ├── fairness/
│   ├── privacy/
│   └── safety/
├── perspectives/
│   ├── safety/
│   │   ├── README.md
│   |   └── ...
│   ├── privacy/
│   │   ├── README.md
│   |   └── ...
│   └── fairness/
│       ├── README.md
│       └── ...
├── chat.py
├── utils.py
├── summarize.py
├── __init__.py
└── README.md
```

---
