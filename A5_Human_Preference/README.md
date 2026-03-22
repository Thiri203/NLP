# DPO Fine-Tuning with Human Preference Data

## Overview

This project fine-tunes a large language model using **Direct Preference
Optimization (DPO)** to align model responses with human preferences.

The base model is **Qwen2.5-1.5B-Instruct**, and it is fine-tuned using
the **truthy-dpo-v0.1 dataset**.

------------------------------------------------------------------------

## Base Model

Qwen/Qwen2.5-1.5B-Instruct

------------------------------------------------------------------------

## Dataset

jondurbin/truthy-dpo-v0.1

Dataset structure: - prompt - chosen - rejected

Dataset size: - Train: 812 - Eval: 204 - Total: 1016

------------------------------------------------------------------------

## Training Method

The model was trained using **Direct Preference Optimization (DPO)**
with LoRA adapters.

Training configuration: - Epochs: 1 - Learning rate: 5e-7 - Batch size:
1 - Max sequence length: 512 - DPO beta: 0.1 - Hardware: Google Colab
(Tesla T4 GPU)

------------------------------------------------------------------------

## Training Results

-   Final Training Loss: 0.6786
-   Final Validation Loss: 0.6698
-   Training Runtime: \~29 minutes

------------------------------------------------------------------------

## Model Repository

Hugging Face Model:
https://huggingface.co/thirishinthant23/A5-NLP_Human_Preference

This repository contains the LoRA adapter weights trained with DPO.

------------------------------------------------------------------------

## Evaluation

Evaluation was performed using prompts from the **AlpacaEval dataset**.

Two models were compared:

Model A -- Base model (Qwen2.5-1.5B-Instruct)\
Model B -- DPO fine-tuned model

Gemini was used as an **LLM-as-a-Judge** to determine which response was
better.

Due to Google Colab runtime limitations and Gemini free-tier API rate
limits, **6 evaluation samples were completed**.

Evaluation results:

  Sample   Winner
  -------- ---------
  1        Model B
  2        Model A
  3        Tie
  4        Model A
  5        Model A
  6        Model A

### Win Rate

WinRate = (wins + 0.5 \* ties) / total

WinRate = (1 + 0.5 × 1) / 6 = **25%**

------------------------------------------------------------------------

## Conclusion

This project demonstrates the workflow for training a language model
using **Direct Preference Optimization (DPO)** and evaluating it with an
**LLM-as-a-Judge framework**.

While the evaluation sample size is limited due to runtime constraints,
the complete training and evaluation pipeline was successfully
implemented.
