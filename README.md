<div align='center'>
<h1>DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios</h1>

Tongzhou Yu, Mingjia Li, Hong Qian*, Jiajun Guo, Wenkai Wang, Zongbao Zhang, Yaoyu Jiang, Xiangfeng Wang, and Aimin Zhou

(*Corresponding author)

East China Normal University, Shanghai Innovation Institute, Nanjing University

<a href='https://anonymous.4open.science/r/DEFINED/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='DEFINED-KDD-2026.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>

<img src='img/methods_model.png' />
</div>

------

## 📰 News
- [x] [2026.05] DEFINED repository released.

# Abstract and Contribution
Human creativity has emerged as a critical competency in the era of large language models. Assessing creativity in complex, open-ended environments is a grand challenge in data mining, currently hindered by a reliance on standardized simple tasks and the scarcity of fine-grained expert data.

As an ecologically valid assessment context, debate reflects multiple dimensions of creativity, encompassing both divergent thinking and convergent thinking. However, current automated scoring methods are poorly suited to complex settings such as debate, and therefore still rely on costly human evaluation.

This repository provides the codebase for **DEFINED**, a **d**ata-**e**fficient computational framework for **f**ine-gra**in**ed cr**e**ativity assessment in **d**ebate scenarios.

- We collect authentic debate competition statements scored by expert adjudicators and adopt a triple-constraint data augmentation strategy to mitigate elite bias.
- We propose an eight-dimensional metric system that supports both fine-grained (8-dim) scoring and coarse-grained holistic scoring.
- We introduce a mixed-granularity training strategy that enables robust learning from limited fine-grained expert supervision.

## Framework
![Framework](./img/methods_model.png)

![Evaluation Protocol](./img/methods_eval.png)

![Coarse-Grained Results](./img/results_pic_v2_1.png)

# Quick Start
## Installation
```bash
pip install -r requirements.txt
```

## Training
Edit the paths in `data_analysis/debate_creativity_rm.sh` (e.g., `MODEL_PATH`, `reward_data_path`, `template_path`, `val_path`), then run:

```bash
cd data_analysis
bash debate_creativity_rm.sh
```

## Inference
Edit the paths in `data_analysis/inference_debate_rm.sh` (e.g., `MODEL_PATH`, `ADAPTER_PATH`, `DATA_PATH`, `OUTPUT_PATH`), then run:

```bash
cd data_analysis
bash inference_debate_rm.sh
```

# Directory Structure
## Data Collection (`data_collection/`)
- `data_extraction.py`: Extracts and segments debate statements from competition records.
- `inference_generate_pair.py`: Generates debate statements of low-to-mid score range (augmentation).
- `inference_summary.py`: Generates summarization variants for contextual variation.

## Data Analysis (`data_analysis/`)
**Benchmark (`data_analysis/benchmark/`)**
- `Debatrix.py` and `Inspiredebate.py`: Baseline methods for coarse-grained scoring comparisons.
- `run_evaluation.py`: Fine-grained scoring by calling external APIs.

**DEFINED Training**
- `debate_creativity_rm.py`: Main training script.
- `debate_rm_creativity_trainer.py`: Trainer and training utilities.
- `accelerate_config_debate_rm.yaml`: Accelerate configuration.

**DEFINED Inference**
- `inference_debate_rm.py`: Inference script for scoring.
- `qwen2.5-7b.jinja`: Prompt template.

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
