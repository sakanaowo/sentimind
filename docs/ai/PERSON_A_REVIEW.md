---
title: Person A — Review & Hướng Dẫn Chạy
description: Tóm tắt những gì đã xây dựng (M1, M2, M3) và hướng dẫn đầy đủ để chạy pipeline
date: 2026-03-11
---

# Person A — Review & Hướng Dẫn Chạy

> Milestone đã hoàn thành: **M1** (khởi tạo dự án) · **M2** (tiền xử lý dữ liệu) · **M3** (BiLSTM baseline)

---

## 1. Tổng Quan Những Gì Đã Xây Dựng

### M1 — Khởi tạo dự án & chuẩn hóa cấu trúc

| File / Thư mục | Vai trò |
|---|---|
| `configs/preprocessing.yaml` | Cấu hình toàn bộ pipeline tiền xử lý (đường dẫn, label map, tỉ lệ split) |
| `configs/bilstm.yaml` | Cấu hình model, training, output cho BiLSTM |
| `data_contract.md` | Hợp đồng dữ liệu — schema cột, label mapping, metric schema |
| `src/__init__.py` + các `__init__` trong sub-package | Khung package Python |
| `data/artifacts/.gitkeep`, `data/raw/.gitkeep` | Giữ thư mục artifacts/raw trong git |

### M2 — Tiền xử lý dữ liệu

| File | Vai trò |
|---|---|
| `src/data/preprocess.py` | Logic làm sạch text + chuẩn hóa label (pure functions, deterministic) |
| `src/data/dataset.py` | `Vocabulary`, `SentimentDataset` (PyTorch), `build_vocab_and_loaders`, `compute_class_weights` |
| `scripts/preprocess.py` | Entry-point: chạy pipeline → xuất `train/val/test.csv` + báo cáo chất lượng |
| `tests/test_preprocessing.py` | Unit tests cho cleaning và split |

**Các bước xử lý tự động khi chạy `preprocess.py`:**

1. Load raw CSV
2. Làm sạch text: Unicode NFKC → bỏ URL → bỏ @mention → giữ chữ hashtag (bỏ `#`) → bỏ HTML entity → bỏ ký tự điều khiển → lowercase
3. Drop: null rows, text quá ngắn (< 3 ký tự), label không hợp lệ, duplicate (text + label_id)
4. Stratified split: 70% train / 15% val / 15% test (seed=42)
5. Lưu `data/processed/train.csv`, `val.csv`, `test.csv`
6. Lưu báo cáo chất lượng `data/artifacts/preprocessing_report.json`
7. Schema validation smoke-check tự động

**Label mapping (7 lớp):**

| Label | ID |
|---|---|
| normal | 0 |
| depression | 1 |
| anxiety | 2 |
| bipolar | 3 |
| ptsd | 4 |
| stress | 5 |
| suicidal | 6 |

### M3 — BiLSTM Baseline

| File | Vai trò |
|---|---|
| `src/models/bilstm.py` | `BiLSTMClassifier`: Embedding → BiLSTM → Mean pooling → Dropout → Linear |
| `src/training/trainer.py` | Training loop: Adam, CrossEntropyLoss (weighted), gradient clipping, early stopping, checkpoint |
| `src/utils/metrics.py` | `compute_metrics`, `save_metrics`, `save_confusion_matrix_plot` |
| `scripts/train_bilstm.py` | Entry-point: build vocab → build model → train → lưu checkpoint + history |
| `scripts/eval_bilstm.py` | Entry-point: load checkpoint → evaluate test/val → in metrics + confusion matrix |
| `tests/test_bilstm_model.py` | Unit tests cho model forward pass và trainer |

**Kiến trúc BiLSTM:**

```
Input (batch, seq_len=128)
  → Embedding (vocab_size, 300d) + Dropout(0.3)
  → BiLSTM (hidden=256 × 2 chiều, 2 layers, dropout=0.3)
  → Mean Pooling (batch, 512)
  → Dropout(0.3) → Linear(512, 7)
  → Logits (batch, 7)
```

**Hyperparameters mặc định** (`configs/bilstm.yaml`):

| Param | Giá trị |
|---|---|
| embedding_dim | 300 |
| hidden_dim | 256 |
| bidirectional | true |
| num_layers | 2 |
| dropout | 0.3 |
| batch_size | 64 |
| epochs | 30 (+ early stopping patience=5) |
| learning_rate | 0.001 |
| weight_decay | 1e-4 |
| gradient_clip | 5.0 |
| class_weighted_loss | true |
| early_stopping_metric | macro_f1 |

---

## 2. Cài Đặt Môi Trường

```bash
# Tạo và kích hoạt virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Cài dependencies
pip install -r requirements.txt
```

---

## 3. Chuẩn Bị Dữ Liệu

Đặt file CSV thô vào `data/raw/`. Dataset mặc định là Kaggle Mental Health dataset:

```
data/raw/kaggle_mental_health.csv
```

File CSV cần có ít nhất 2 cột: `text` và `label`.  
Tên cột và đường dẫn có thể cấu hình lại trong `configs/preprocessing.yaml`.

---

## 4. Hướng Dẫn Chạy Từng Bước

### Bước 1 — Tiền xử lý dữ liệu

```bash
# Chạy với config mặc định
python scripts/preprocess.py

# Override đường dẫn raw data
python scripts/preprocess.py --raw_path data/raw/my_data.csv

# Dùng config khác
python scripts/preprocess.py --config configs/preprocessing.yaml
```

**Output:**

```
data/processed/
  train.csv          ← 70% dữ liệu, stratified
  val.csv            ← 15%
  test.csv           ← 15%
data/artifacts/
  preprocessing_report.json   ← thống kê chất lượng dữ liệu
```

Ví dụ log thành công:

```
10:00:01 | INFO | Splits saved to data/processed.
10:00:01 | INFO | ✓ All split files pass schema validation.
```

---

### Bước 2 — Huấn luyện BiLSTM

```bash
# Chạy với config mặc định (CPU)
python scripts/train_bilstm.py

# Dùng GPU
python scripts/train_bilstm.py --device cuda

# Chỉ định GPU cụ thể
python scripts/train_bilstm.py --device cuda:0

# Dùng config khác
python scripts/train_bilstm.py --config configs/bilstm.yaml
```

**Output:**

```
data/artifacts/
  bilstm_best.pt              ← checkpoint tốt nhất (theo macro_f1)
  bilstm_train_history.json   ← loss/acc/f1 theo từng epoch
  vocab.json                  ← vocabulary đã build từ train set
```

Ví dụ log epoch:

```
10:05:23 | INFO | Epoch   1/30 | train_loss=1.7823 acc=0.3412 | val_loss=1.6102 acc=0.4201 macro_f1=0.3894 | 42.3s
10:05:23 | INFO |   ✓ New best checkpoint saved (epoch 1, macro_f1=0.3894).
```

Training dừng sớm nếu `macro_f1` không cải thiện sau 5 epochs liên tiếp.

---

### Bước 3 — Đánh giá BiLSTM

```bash
# Đánh giá trên test set (mặc định)
python scripts/eval_bilstm.py

# Đánh giá trên val set
python scripts/eval_bilstm.py --split val

# Chỉ định device
python scripts/eval_bilstm.py --device cuda
```

**Output:**

```
data/artifacts/
  bilstm_metrics.json              ← accuracy, macro_f1, weighted_f1, per-class metrics
  bilstm_confusion_matrix.png      ← confusion matrix dạng hình ảnh
```

Ví dụ in ra màn hình:

```
=======================================================
  BiLSTM Evaluation Results (test split)
=======================================================
  Accuracy  : 0.8234
  Macro F1  : 0.8101
  Weighted F1: 0.8219
=======================================================
  Normal       P=0.851 R=0.834 F1=0.842  n=432
  Depression   P=0.801 R=0.813 F1=0.807  n=389
  ...
=======================================================
```

---

### Bước 4 — Chạy Tests

```bash
# Chạy toàn bộ test suite
python -m pytest tests/ -v

# Chỉ test preprocessing
python -m pytest tests/test_preprocessing.py -v

# Chỉ test model
python -m pytest tests/test_bilstm_model.py -v
```

---

## 5. Toàn Bộ Pipeline Một Lệnh

```bash
python scripts/preprocess.py && python scripts/train_bilstm.py && python scripts/eval_bilstm.py
```

---

## 6. Cấu Hình Nâng Cao

### Dùng GloVe Twitter embeddings (tùy chọn)

Tải GloVe Twitter 300d từ [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove) và đặt vào `data/raw/`. Sau đó cập nhật `configs/bilstm.yaml`:

```yaml
model:
  pretrained_embeddings: data/raw/glove.twitter.27B.300d.txt
  freeze_embeddings: false   # true = đóng băng embedding khi train
```

### Đổi label mapping

Sửa `configs/preprocessing.yaml` → phần `label_map`. Ví dụ nếu dataset chỉ có 3 nhãn:

```yaml
label_map:
  negative: 0
  neutral: 1
  positive: 2
```

### Early stopping theo val_loss thay vì macro_f1

```yaml
training:
  early_stopping_metric: val_loss
```

---

## 7. Sơ Đồ Luồng Dữ Liệu

```
data/raw/kaggle_mental_health.csv
        │
        ▼
scripts/preprocess.py
  ├─ src/data/preprocess.py  (clean_text, normalise_label, preprocess_dataframe)
  │
  ├─ data/processed/train.csv  (70%)
  ├─ data/processed/val.csv    (15%)
  ├─ data/processed/test.csv   (15%)
  └─ data/artifacts/preprocessing_report.json
        │
        ▼
scripts/train_bilstm.py
  ├─ src/data/dataset.py       (Vocabulary, SentimentDataset, DataLoader)
  ├─ src/models/bilstm.py      (BiLSTMClassifier)
  ├─ src/training/trainer.py   (train loop, EarlyStopping)
  │
  ├─ data/artifacts/vocab.json
  ├─ data/artifacts/bilstm_best.pt
  └─ data/artifacts/bilstm_train_history.json
        │
        ▼
scripts/eval_bilstm.py
  ├─ src/utils/metrics.py      (compute_metrics, save_confusion_matrix_plot)
  │
  ├─ data/artifacts/bilstm_metrics.json
  └─ data/artifacts/bilstm_confusion_matrix.png
```

---

## 8. Artifacts Đầu Ra Tóm Tắt

| File | Vị trí | Mô tả |
|---|---|---|
| `train.csv` / `val.csv` / `test.csv` | `data/processed/` | Dataset đã xử lý, sẵn sàng cho mọi model |
| `preprocessing_report.json` | `data/artifacts/` | Thống kê chất lượng dữ liệu, class distribution |
| `vocab.json` | `data/artifacts/` | Vocabulary build từ train set |
| `bilstm_best.pt` | `data/artifacts/` | Best checkpoint (state_dict) |
| `bilstm_train_history.json` | `data/artifacts/` | Lịch sử train theo epoch |
| `bilstm_metrics.json` | `data/artifacts/` | Tất cả metrics theo schema contract |
| `bilstm_confusion_matrix.png` | `data/artifacts/` | Confusion matrix dạng ảnh |

---

## 9. Lưu Ý Khi Bàn Giao Cho Person B

- `data/processed/*.csv` là input chuẩn cho **M4 (BERTweet)**, **M5 (LLM)**, **M6 (Semantic)**. Schema đảm bảo: cột `text`, `label`, `label_id`.
- `data/artifacts/bilstm_metrics.json` tuân thủ metric schema trong `data_contract.md`. Person B cần xuất metrics BERTweet/LLM theo cùng schema này để M7 so sánh.
- `configs/preprocessing.yaml` → `label_map` là nguồn truth duy nhất cho label ID mapping.
- Seed cố định `42` trong toàn bộ pipeline → kết quả tái lập được.
