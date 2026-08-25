# Trustworthy_DLLM

Trustworthy_DLLM is a benchmark and evaluation toolkit for studying the **trustworthiness of diffusion language models** across three perspectives:

- safety
- privacy
- fairness

The repository contains two complementary evaluation pipelines:

- `trustldm_static/`: fixed benchmark evaluation with predefined datasets, prompts, and metrics
- `trustldm_auto/`: automatic attack search and adaptive evaluation pipeline

This work has been accepted to **EMNLP 2026**. The public paper is available as
[TrustLDM: Benchmarking Trustworthiness in Language Diffusion Models](https://arxiv.org/abs/2606.00023).

## Overview

The benchmark focuses on whether diffusion language models remain trustworthy under both standard and adversarial settings:

- **Safety**: whether the model resists harmful completions under contextual wrapping
- **Privacy**: whether the model leaks sensitive information in privacy-aware scenarios
- **Fairness**: whether the model exhibits demographic bias under contextual perturbations

## Pipelines

### `trustldm_static`
`trustldm_static` is the fixed evaluation suite. It runs safety, privacy, and fairness benchmarks on curated datasets and saves perspective-specific results.

![trustldm_static pipeline](./trustldm_static/pipeline.png)

See [trustldm_static/README.md](trustldm_static/README.md) for environment setup, configs, and per-perspective usage.

### `trustldm_auto`
`trustldm_auto` is the automatic evaluation pipeline. It iteratively generates `pre_context` and `post_context` wrappers, screens candidate outputs, and refines higher-scoring attacks.

![trustldm_auto pipeline](./trustldm_auto/pipeline.png)

See [trustldm_auto/README.md](trustldm_auto/README.md) for setup and usage.

## Target Models

### Open-Source Models
- **LLaDA-8B-Instruct**
- **LLaDA-1.5**
- **LLaDA-MoE-7B-A1B-Instruct**
- **Dream-v0-Instruct-7B**

The repository also keeps `generate_functions/generate_fastdllmv2.py` as an
extension example. Additional diffusion model generators can be added without
changing the evaluation interfaces.

### Closed-Source Models
- **Mercury** (legacy / optional code path still kept in the repository)

## Getting Started

Detailed installation and environment setup differ between `trustldm_static` and `trustldm_auto`, so please follow the README inside the pipeline you plan to run:

- For fixed benchmark evaluation: [trustldm_static/README.md](trustldm_static/README.md)
- For automatic attack evaluation: [trustldm_auto/README.md](trustldm_auto/README.md)

### PyTorch and CUDA environments

Choose the environment according to the GPU architecture:

| GPU/software stack | PyTorch | Python | Intended use |
|---|---|---|---|
| Legacy CUDA 11.8 setup | `torch==2.0.1` or `torch==2.1.0+cu118` | 3.10 | Older GPUs and the historical model-specific environments |
| Newer GPU setup | `torch==2.7.1+cu128` | 3.12 | Current newer-architecture GPUs; one environment for LLaDA and Dream |

For the newer setup, create the maintained `trustldm-d2f` environment and
install the CUDA 12.8 PyTorch wheel first:

```bash
conda create -n trustldm-d2f python=3.12 -y
conda activate trustldm-d2f
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-trustldm-d2f.txt
```

The validated newer stack also uses `transformers==4.57.6` and
`tokenizers==0.22.2`, and supports the LLaDA family and Dream in one
environment. See [requirements-trustldm-d2f.txt](./requirements-trustldm-d2f.txt)
and the pipeline-specific READMEs for the remaining setup steps.

### Common Dependency Note

Older releases used separate environments because of the model-dependent `transformers` / `tokenizers` version split:

- **LLaDA-family models** typically use `transformers==4.43.2` and `tokenizers==0.19.1`
- **Dream** typically uses `transformers==4.47.1` and `tokenizers==0.21.4`

The newer `trustldm-d2f` stack removes this inconvenience on supported newer GPUs.

If you plan to evaluate newer model families beyond the ones documented here,
check their CUDA and PyTorch requirements separately. The `trustldm-d2f`
environment is the maintained starting point for newer GPU architectures.

## Repository Structure

```text
.
├── data/                  # benchmark data
├── generate_functions/    # generation helpers for supported models
├── requirements-trustldm-d2f.txt  # unified newer-GPU environment pins
├── scripts/               # utility scripts
├── trustldm_static/       # static benchmark pipeline
├── trustldm_auto/         # automatic evaluation pipeline
├── setup.cfg              # package metadata and trustldm-stat CLI entry point
└── README.md              # repository overview
```

## Notes

- Several config files contain placeholder absolute model paths. Update them to your local checkpoints before running experiments.
- Perspective-specific dataset-field mappings and paper-name mappings are documented in the README files under `trustldm_static/perspectives/`.

## Citation

```bibtex
@misc{mo2026trustldmbenchmarkingtrustworthinesslanguage,
  title={TrustLDM: Benchmarking Trustworthiness in Language Diffusion Models},
  author={Yichuan Mo and Yukun Jiang and Yanbo Shi and Mingjie Li and Michael Backes and Yang Zhang and Yisen Wang},
  year={2026},
  eprint={2606.00023},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2606.00023}
}
```
