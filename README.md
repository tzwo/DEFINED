<div align='center'>
<h1>DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios</h1>

Tongzhou Yu*, Mingjia Li*, Hong Qian, Jiajun Guo, Wenkai Wang, Zongbao Zhang, Yaoyu Jiang, Xiangfeng Wang, and Aimin Zhou

*Equal contribution. Hong Qian is the corresponding author.

Nanjing University, East China Normal University, Shanghai Innovation Institute

<a href='https://anonymous.4open.science/r/DEFINED/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='DEFINED-KDD-2026.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
</div>

------

:sparkles: Welcome to **DEFINED**, a comprehensive repository for **fine-grained creativity assessment in debate scenarios**. This project studies how to move beyond conventional creativity tests and build an ecologically valid, data-efficient scoring framework from authentic debate data.

## 📰 News
- [x] [2026.05] DEFINED repository released.

![DEFINED Framework](./img/methods_model.png)

# Abstract and Contribution

Human creativity has become a critical competency in the era of large language models. Yet, assessing creativity in complex and open-ended environments remains difficult because most existing approaches rely on simplified tasks and large amounts of expensive expert annotations.

Debate provides a naturally rich setting for creativity assessment: it combines **divergent thinking** with **convergent thinking**, requires contextual reasoning, and reflects realistic human judgment under adversarial interaction. However, current automated scoring methods still struggle in such complex settings and often depend on costly human evaluation.

To address this challenge, we propose **DEFINED**, a **d**ata-**e**fficient computational framework for **f**ine-gra**in**ed cr**e**ativity assessment in **d**ebate scenarios.

- **Authentic debate data + triple-constraint augmentation.** We collect real competition statements scored by expert adjudicators and augment them to alleviate elite-data bias.
- **Eight-dimensional metric system.** We model debate creativity through five creativity-related dimensions and three debate-related dimensions, enabling both fine-grained and coarse-grained evaluation.
- **Mixed-granularity training.** We learn from limited fine-grained annotations together with a much larger set of coarse-grained supervision signals.
- **Human-aligned scoring.** DEFINED is designed to approximate the cognitive process of expert adjudicators rather than only predicting a single holistic score.

# Framework Overview

The overall framework contains two tightly coupled components:

- **Data construction and augmentation.** We build a mixed-granularity dataset from authentic competitions, synthetic mid-to-low proficiency data, negative samples, and summarization variants.
- **Scoring model.** We use a pre-trained autoregressive language model as the semantic encoder and a hierarchical scoring head to predict eight dimension scores and an overall debate score.

![Framework Overview](./img/methods_model.png)

In DEFINED, the eight dimensions include:

- **Creativity-specific dimensions:** Fluency, Originality, Flexibility, Logicality, and Relevance
- **Non-creativity dimensions:** Effectiveness, Clarity, and Appeal

This decomposition allows the model to perform fine-grained creativity assessment while preserving compatibility with coarse-grained overall scoring.

# Evaluation Protocol

To validate both ecological validity and scoring robustness, the paper introduces a **three-modular evaluation protocol**:

- **Fine-grained evaluation:** compares dimension-wise predictions against expert annotations using MSE and PCC.
- **Coarse-grained evaluation on mid-to-low proficiency data:** uses pairwise agreement to assess ranking robustness.
- **Coarse-grained evaluation on authentic high-proficiency data:** uses MSE against top-tier adjudicator scores.

![Evaluation Protocol](./img/methods_eval.png)

This protocol is important because creativity assessment in debate must work across both **different proficiency levels** and **different annotation granularities**.

# Main Results

DEFINED achieves strong performance across all three evaluation modules:

- **Fine-grained scoring:** average PCC reaches **0.96**, with an average MSE of **43.09**.
- **Mid-to-low proficiency ranking:** pairwise accuracy reaches **95.2%**, substantially outperforming debate-evaluation baselines.
- **High-proficiency scoring:** DEFINED achieves an MSE of **18.23**, much closer to expert adjudicators than prompt-based baselines.

## Fine-Grained Results

The table below reports the **Pearson correlation coefficient (PCC)** between model predictions and expert scores on the eight fine-grained dimensions.

| Model | Fluency | Originality | Flexibility | Logicality | Relevance | Effectiveness | Clarity | Appeal | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini-2.5-pro | 0.71 | 0.77 | 0.76 | 0.71 | 0.67 | 0.71 | 0.44 | 0.61 | 0.67 |
| GPT-4o | 0.70 | 0.64 | 0.67 | 0.68 | 0.70 | 0.70 | 0.58 | 0.71 | 0.67 |
| Qwen3-max-preview | 0.74 | 0.77 | 0.75 | 0.69 | 0.75 | 0.75 | 0.50 | 0.74 | 0.71 |
| Deepseek-R1 | 0.79 | 0.77 | 0.74 | 0.74 | 0.77 | 0.83 | 0.55 | 0.76 | 0.74 |
| M-Prometheus-7B | 0.59 | 0.59 | 0.53 | 0.60 | 0.57 | 0.62 | 0.56 | 0.55 | 0.58 |
| Themis | -0.02 | 0.31 | 0.20 | 0.07 | 0.16 | 0.37 | -0.22 | 0.14 | 0.12 |
| **DEFINED** | **0.96** | **0.96** | **0.94** | **0.96** | **0.96** | **0.96** | **0.93** | **0.97** | **0.96** |

We also report the **mean squared error (MSE)** of fine-grained score prediction:

| Model | Fluency | Originality | Flexibility | Logicality | Relevance | Effectiveness | Clarity | Appeal | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini-2.5-pro | 358.75 | 465.68 | 444.08 | 419.44 | 506.66 | 555.39 | 597.76 | 550.12 | 487.24 |
| GPT-4o | 305.51 | 347.46 | 400.42 | 372.10 | 355.73 | 369.64 | 395.76 | 332.93 | 359.94 |
| Qwen3-max-preview | 243.64 | 208.93 | 251.24 | 300.58 | 231.15 | 230.19 | 440.25 | 276.44 | 272.80 |
| Deepseek-R1 | 298.73 | 207.10 | 226.75 | 240.95 | 225.88 | 166.90 | 332.37 | 271.61 | 246.29 |
| M-Prometheus-7B | 352.50 | 402.72 | 465.48 | 405.02 | 488.47 | 416.18 | 407.10 | 510.42 | 430.99 |
| Themis | 1943.67 | 1143.48 | 1148.60 | 1830.02 | 1576.83 | 963.40 | 2366.62 | 1483.35 | 1557.00 |
| **DEFINED** | **42.24** | **34.04** | **59.60** | **46.72** | **35.98** | **35.76** | **57.01** | **33.36** | **43.09** |

![Coarse-Grained Results](./img/results_pic_v2_1.png)

These results indicate that DEFINED not only predicts scores accurately, but also better aligns with the underlying cognitive structure of human debate evaluation.

# Quick Start

## Installation

```bash
pip install -r requirements.txt
```

## Training

Edit the paths in `data_analysis/debate_creativity_rm.sh` first, including:

- `MODEL_PATH`
- `reward_data_path`
- `template_path`
- `val_path`

Then run:

```bash
cd data_analysis
bash debate_creativity_rm.sh
```

## Inference

Edit the paths in `data_analysis/inference_debate_rm.sh` first, including:

- `MODEL_PATH`
- `ADAPTER_PATH`
- `DATA_PATH`
- `OUTPUT_PATH`

Then run:

```bash
cd data_analysis
bash inference_debate_rm.sh
```

# Repository Structure

## Data Collection (`data_collection/`)

- `data_extraction.py`: extracts and segments debate statements from competition records.
- `inference_generate_pair.py`: generates augmented debate statements for the low-to-mid score range.
- `inference_summary.py`: generates summarization variants for contextual variation.

## Data Analysis (`data_analysis/`)

**Benchmark (`data_analysis/benchmark/`)**

- `Debatrix.py` and `Inspiredebate.py`: baseline methods for coarse-grained scoring comparisons.
- `run_evaluation.py`: fine-grained scoring by calling external APIs.

**DEFINED Training**

- `debate_creativity_rm.py`: main training script.
- `debate_rm_creativity_trainer.py`: trainer and training utilities.
- `accelerate_config_debate_rm.yaml`: accelerate configuration.

**DEFINED Inference**

- `inference_debate_rm.py`: inference script for scoring.
- `inference_debate_rm.sh`: shell entry for inference.
- `qwen2.5-7b.jinja`: prompt template.

# Reference

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{DEFINED2026kdd,
  title     = {DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios},
  author    = {Yu, Tongzhou and Li, Mingjia and Qian, Hong and Guo, Jiajun and Wang, Wenkai and Zhang, Zongbao and Jiang, Yaoyu and Wang, Xiangfeng and Zhou, Aimin},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  doi       = {10.1145/3770855.3817874}
}
```
