# Project Overview

This repository contains the codebase for data collection, processing, and analysis for DEFINED. It is organized into two main directories: `data_collection` for data handling and `data_analysis` for model training and evaluation.

## Directory Structure & Usage

### 1. Data Collection (`data_collection/`)

This folder contains scripts responsible for data processing and generation.

- **`data_extraction.py`**
  - **Function**: Extracts and segments debate statements from competition records.
  - **Usage**: Use this to process raw match data into structured segments.

- **`inference_generate_pair.py`**
  - **Function**: Generates debate statements of low-to-mid score range.
  - **Usage**: Utilized for creating data with specific quality attributes.

- **`inference_summary.py`**
  - **Function**: Generates summary variants of the input texts.
  - **Usage**: Run this to produce summary variants.

### 2. Data Analysis (`data_analysis/`)

This folder handles the training of the DEFINED model, benchmarking, and inference.

#### Benchmark (`data_analysis/benchmark/`)

Contains baseline methods and evaluation tools.

- **`Debatrix.py`** & **`Inspiredebate.py`**
  - **Function**: Two baseline methods used for coarse-grained scoring comparisons.

- **`run_evaluation.py`**
  - **Function**: Performs fine-grained scoring by calling external APIs.
  - **Usage**: Use this script for fine-grained evaluation metrics.

#### DEFINED Model Training

Scripts related to training the model.

- **`debate_creativity_rm.py`**
  - **Function**: The main training script for the DEFINED model.
- **`debate_rm_creativity_trainer.py`**
  - **Function**: Helper module containing training logic and classes.
- **`debate_creativity_rm.sh`**
  - **Function**: Shell script to launch the training process.
  - **Usage**: Run this script to start training. Ensure configuration parameters (like model paths) are set correctly inside.

#### DEFINED Model Inference

Scripts for applying the trained model.

- **`inference_debate_rm.py`**
  - **Function**: Python script for running inference using the trained model.
- **`inference_debate_rm.sh`**
  - **Function**: Shell script to execute the inference pipeline.
  - **Usage**: Run this to obtain scores on new data using the trained DEFINED model.
