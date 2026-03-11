---
phase: requirements
title: Requirements & Problem Understanding
description: Clarify the problem space, gather requirements, and define success criteria
---

# Requirements & Problem Understanding

## Problem Statement
**What problem are we solving?**

- We need a lightweight RNN baseline to benchmark against Transformer and LLM approaches.
- Users are model developers and evaluators comparing quality/cost trade-offs.
- Current situation lacks a reproducible baseline with clear hyperparameter documentation.

## Goals & Objectives
**What do we want to achieve?**

- Implement BiLSTM classifier: Embedding -> BiLSTM -> Dropout -> Dense -> Softmax.
- Support pretrained embeddings option (GloVe/word2vec-compatible) and random init fallback.
- Provide reproducible training and evaluation metrics with confusion matrix.
- Non-goals: SOTA optimization, complex attention variants, distributed training.

## User Stories & Use Cases
**How will users interact with the solution?**

- As a researcher, I want to train BiLSTM quickly so I can establish a baseline in one run.
- As a reviewer, I want saved config + seed + metrics so I can reproduce the result.
- As an analyst, I want per-class F1 to inspect weaknesses on minority labels.
- Edge cases: OOV tokens, very short/very long text truncation effects.

## Success Criteria
**How will we know when we're done?**

- Training script runs end-to-end on preprocessed data and exports model + metrics.
- Baseline target reaches planned range (~0.75-0.80 accuracy, subject to data quality).
- Evaluation outputs include macro/micro F1 and confusion matrix artifact.
- Acceptance: experiment is reproducible from documented commands only.

## Constraints & Assumptions
**What limitations do we need to work within?**

- Technical constraints: moderate compute, prefer PyTorch implementation for consistency.
- Time constraints: baseline should train in <2 hours on available hardware.
- Assumptions: tokenization approach is stable enough for RNN baseline.
- Cost constraints: no external API dependency.

## Questions & Open Items
**What do we still need to clarify?**

- Which embedding dimension and max sequence length should be default?
- Early stopping metric: validation macro F1 or loss?
- Need class-weighted loss by default or configurable?
- Should we include ablation for pretrained vs random embeddings in first cycle?
