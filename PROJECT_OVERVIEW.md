# Sentimind Project Overview

This repository currently focuses on two supervised models for mental-health sentiment classification:
- BiLSTM baseline
- BERTweet transformer

LLM-based prompting modules were removed from this codebase.

## Current structure (high level)
- configs/: preprocessing, bilstm, bertweet, semantic configs
- scripts/: preprocess, train/eval bilstm, train/eval bertweet, semantic analysis
- src/: data pipeline, model definitions, training utilities, metrics
- tests/: preprocessing, bilstm, bertweet, semantic tests
- notebooks/: analysis and reporting notebooks

## Main artifacts
- data/artifacts/bilstm_*.pt / *.json / confusion matrix
- data/artifacts/bertweet_*.pt / *.json / confusion matrix
- data/artifacts/sts_report.json
- data/artifacts/semantic_embeddings.npy
- data/artifacts/semantic_cluster_plot.png
- data/artifacts/comparison_report.json
