# Reproduce Guide

This guide reproduces the current supervised Sentimind pipeline.

## Environment
- Python 3.10+
- Install dependencies:
  - pip install -r requirements.txt

## Pipeline order
1. Preprocess data
   - python scripts/preprocess.py
2. Train BiLSTM
   - python scripts/train_bilstm.py
3. Evaluate BiLSTM
   - python scripts/eval_bilstm.py
4. Train BERTweet
   - python scripts/train_bertweet.py
5. Evaluate BERTweet
   - python scripts/eval_bertweet.py
6. Run semantic analysis
   - python scripts/run_semantic_analysis.py

## Tests
- pytest -q

## Notes
- LLM prompting components were removed from this repository.
