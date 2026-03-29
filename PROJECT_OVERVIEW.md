# Sentimind — Tổng Quan Dự Án

> **Tài liệu trình bày toàn diện** về dự án phân tích cảm xúc sức khỏe tâm thần, so sánh ba kiến trúc mô hình: RNN (BiLSTM) vs Transformer (BERTweet) vs LLM (Gemini).

---

## Mục Lục

1. [Giới Thiệu Bài Toán](#1-giới-thiệu-bài-toán)
2. [Câu Hỏi Nghiên Cứu](#2-câu-hỏi-nghiên-cứu)
3. [Dữ Liệu](#3-dữ-liệu)
4. [Pipeline Tổng Thể](#4-pipeline-tổng-thể)
5. [Tiền Xử Lý Dữ Liệu (M2)](#5-tiền-xử-lý-dữ-liệu-m2)
6. [Model 1 — BiLSTM (M3)](#6-model-1--bilstm-m3)
7. [Model 2 — BERTweet (M4)](#7-model-2--bertweet-m4)
8. [Model 3 — LLM Prompting (M5)](#8-model-3--llm-prompting-m5)
9. [Phân Tích Ngữ Nghĩa — Semantic Analysis (M6)](#9-phân-tích-ngữ-nghĩa--semantic-analysis-m6)
10. [Kết Quả & So Sánh](#10-kết-quả--so-sánh)
11. [Cấu Trúc Dự Án](#11-cấu-trúc-dự-án)
12. [Cách Chạy Lại Toàn Bộ Pipeline](#12-cách-chạy-lại-toàn-bộ-pipeline)
13. [Điểm Mạnh, Hạn Chế & Hướng Mở Rộng](#13-điểm-mạnh-hạn-chế--hướng-mở-rộng)

---

## 1. Giới Thiệu Bài Toán

### Bối cảnh

Mạng xã hội (Twitter, Reddit) là nơi người dùng thường xuyên bộc lộ trạng thái tâm lý thông qua những đoạn text ngắn. Những biểu hiện ngôn ngữ này chứa đựng các tín hiệu tinh tế về sức khỏe tâm thần — từ trầm cảm, lo âu, đến rối loạn lưỡng cực hay tư tưởng tự tử. Bài toán đặt ra là: **làm sao một hệ thống máy tính có thể hiểu được các tín hiệu ngữ nghĩa phức tạp này?**

### Tên Dự Án

**Sentimind** — kết hợp giữa _Sentiment_ (cảm xúc) và _Mind_ (tâm trí), thể hiện mục tiêu phân tích cảm xúc trong lĩnh vực sức khỏe tâm thần.

### Phạm Vi

Dự án xây dựng một hệ thống end-to-end bao gồm:

- Tiền xử lý dữ liệu thô (social media posts)
- Huấn luyện và đánh giá **3 kiến trúc mô hình** thuộc 3 thế hệ NLP khác nhau
- Phân tích ngữ nghĩa chiều sâu (Semantic Analysis)
- So sánh toàn diện theo cùng một tiêu chuẩn đánh giá thống nhất

---

## 2. Câu Hỏi Nghiên Cứu

> _"Trong bài toán phân loại cảm xúc sức khỏe tâm thần, mô hình nào (RNN / Transformer / LLM) hiểu ngữ nghĩa tốt hơn và tại sao?"_

Câu hỏi này được trả lời qua 4 góc độ:

1. **Độ chính xác định lượng** — Accuracy, Precision, Recall, F1 trên cùng test set
2. **Ma trận nhầm lẫn** — Phân tích pattern lỗi từng class
3. **Phân tích lỗi** — Sarcasm, phủ định, nhầm lẫn giữa class gần nhau
4. **Chất lượng hiểu ngữ nghĩa** — STS scoring + embedding clustering

---

## 3. Dữ Liệu

### Dataset Chính — Kaggle Mental Health Sentiment

| Thuộc tính  | Giá trị                                                                                                                           |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Nguồn       | [Kaggle: Sentiment Analysis for Mental Health](https://kaggle.com/datasets/suchintikasarkat/sentiment-analysis-for-mental-health) |
| Tên file    | `data/raw/Combined Data.csv`                                                                                                      |
| Kích thước  | ~31 MB, **53,043 dòng** gốc                                                                                                       |
| Cột dữ liệu | `text` (social media posts), `label` (nhãn tình trạng)                                                                            |

### Phân Bố Nhãn (sau tiền xử lý)

| ID       | Nhãn                 | Số mẫu     | Tỉ lệ |
| -------- | -------------------- | ---------- | ----- |
| 0        | Normal               | 16,007     | 31.4% |
| 1        | Depression           | 15,093     | 29.6% |
| 2        | Anxiety              | 3,620      | 7.1%  |
| 3        | Bipolar              | 2,501      | 4.9%  |
| 4        | Personality Disorder | 894        | 1.8%  |
| 5        | Stress               | 2,292      | 4.5%  |
| 6        | Suicidal             | 10,638     | 20.8% |
| **Tổng** |                      | **51,045** | 100%  |

> **Lưu ý:** Dữ liệu **không cân bằng** — Normal và Depression chiếm ~60%, trong khi Personality Disorder chỉ có ~900 mẫu. Đây là một thách thức thực tế cần xử lý.

### Dataset Phụ — TweetEval

- Nguồn: HuggingFace `cardiffnlp/tweet_eval`, subset `sentiment`
- Nhãn: `negative / neutral / positive`
- Vai trò: Tham khảo cho pre-training / transfer learning (không đưa vào split đánh giá)

### Phân Chia Dữ Liệu

| Split      | Tỉ lệ | Số mẫu | Phương pháp                |
| ---------- | ----- | ------ | -------------------------- |
| Train      | 70%   | 35,731 | Stratified (theo label_id) |
| Validation | 15%   | 7,657  | Stratified                 |
| Test       | 15%   | 7,657  | Stratified                 |

> **Seed cố định:** `42` cho tất cả các phép chia — đảm bảo tái lập được hoàn toàn.

---

## 4. Pipeline Tổng Thể

```
┌──────────────────────────────────────────────────┐
│              DỮ LIỆU THÔ                         │
│   data/raw/Combined Data.csv (53,043 dòng)        │
└───────────────────┬──────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│           TIỀN XỬ LÝ (M2)                        │
│  - Unicode NFKC normalisation                     │
│  - Xóa URL, @mention, HTML entities              │
│  - Giữ hashtag word (bỏ dấu #)                   │
│  - Lowercase, drop duplicate & short texts        │
│  - Stratified split 70/15/15 (seed=42)           │
│  → 51,045 mẫu sạch                               │
└───────────────────┬──────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│         PHÂN TÍCH NGỮ NGHĨA (M6)                 │
│  - Sentence embeddings (all-MiniLM-L6-v2, 384D)  │
│  - STS scoring (cosine similarity)               │
│  - UMAP reduction → HDBSCAN clustering           │
└───────────────────┬──────────────────────────────┘
                    ↓
       ┌────────────┼────────────┐
       ↓            ↓            ↓
  BiLSTM (M3)  BERTweet (M4)  LLM (M5)
  [RNN]       [Transformer]  [Gemini API]
       ↓            ↓            ↓
       └────────────┼────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│         ĐÁNH GIÁ & SO SÁNH (M7)                  │
│  - Accuracy, Macro/Weighted F1                   │
│  - Confusion matrix                              │
│  - Per-class Precision / Recall / F1             │
│  - Leaderboard tổng hợp                          │
└──────────────────────────────────────────────────┘
```

---

## 5. Tiền Xử Lý Dữ Liệu (M2)

### Quy Trình Làm Sạch Text (thứ tự không đổi)

1. **Unicode NFKC normalisation** — chuẩn hóa ký tự đặc biệt
2. **Xóa URL** — loại bỏ `http://`, `https://`, `www.`
3. **Xóa @mention** — loại bỏ tag người dùng
4. **Giữ hashtag word** — `#mentalhealth` → `mentalhealth` (bỏ `#`, giữ chữ)
5. **Xóa HTML entities** — `&amp;` → `&`, `&lt;` → `<`, v.v.
6. **Xóa ký tự điều khiển** — control characters không in được
7. **Collapse whitespace** — nhiều khoảng trắng thành 1, strip hai đầu
8. **Lowercase toàn bộ**

### Quy Tắc Loại Bỏ

| Điều kiện                    | Số dòng bị drop |
| ---------------------------- | --------------- |
| Null rows                    | 362             |
| Text quá ngắn (< 3 ký tự)    | 6               |
| Nhãn không hợp lệ            | 0               |
| Duplicates (text + label_id) | 1,630           |
| **Tổng bị drop**             | **1,998**       |

### Kết Quả Tiền Xử Lý

```
Raw:   53,043 dòng
Sau:   51,045 dòng (-1,998 dòng, tỉ lệ giữ lại: 96.2%)
```

### Vocabulary

File `data/artifacts/vocab.json` lưu mapping `{word → index}` dùng cho BiLSTM. Các token đặc biệt: `<PAD>=0`, `<UNK>=1`. Chỉ từ xuất hiện ≥ 2 lần mới được đưa vào từ điển.

### Artifacts Đầu Ra

| File                                       | Mô tả                       |
| ------------------------------------------ | --------------------------- |
| `data/processed/train.csv`                 | 35,731 mẫu train            |
| `data/processed/val.csv`                   | 7,657 mẫu validation        |
| `data/processed/test.csv`                  | 7,657 mẫu test              |
| `data/artifacts/preprocessing_report.json` | Báo cáo chất lượng chi tiết |
| `data/artifacts/vocab.json`                | Từ điển BiLSTM              |

### Code Chính

```
src/data/preprocess.py   → clean_text(), normalise_label(), preprocess_dataframe()
src/data/dataset.py      → Vocabulary, SentimentDataset (PyTorch), build_vocab_and_loaders()
scripts/preprocess.py    → Entry-point CLI
configs/preprocessing.yaml → Tất cả tham số cấu hình
```

---

## 6. Model 1 — BiLSTM (M3)

### Đại Diện Cho: Kiến Trúc RNN Truyền Thống

BiLSTM (Bidirectional Long Short-Term Memory) là mô hình RNN hai chiều — đọc câu từ trái sang phải **và** từ phải sang trái đồng thời, giúp nắm bắt context tốt hơn LSTM một chiều.

### Kiến Trúc

```
Input (batch, seq_len=128)
    ↓
Embedding(vocab_size, 300d) + Dropout(0.3)
    ↓
BiLSTM(hidden=256, 2 layers, bidirectional, dropout=0.3)
    [output: batch × seq_len × 512]
    ↓
Mean Pooling (over sequence) → (batch × 512)
    ↓
Dropout(0.3) → Linear(512 → 7)
    ↓
Logits (batch × 7)
```

**Lý do dùng Mean Pooling** thay vì last hidden state: robust hơn trên văn bản độ dài biến thiên, tránh thông tin ở đầu câu bị "quên" bởi LSTM.

### Hyperparameters

| Tham số               | Giá trị                     | Lý do                        |
| --------------------- | --------------------------- | ---------------------------- |
| `embedding_dim`       | 300                         | Tương thích GloVe/Word2Vec   |
| `hidden_dim`          | 256 (× 2 chiều = 512)       | Cân bằng capacity/tốc độ     |
| `num_layers`          | 2                           | Độ sâu vừa đủ                |
| `dropout`             | 0.3                         | Regularisation               |
| `batch_size`          | 64                          | Phù hợp RAM CPU              |
| `learning_rate`       | 0.001                       | Adam default                 |
| `gradient_clip`       | 5.0                         | Tránh exploding gradient     |
| `early_stopping`      | patience=5, metric=macro_f1 | Tránh overfit majority class |
| `class_weighted_loss` | true                        | Xử lý mất cân bằng nhãn      |
| `max_seq_len`         | 128 tokens                  | Bao phủ 95th-percentile      |

### Xử Lý Mất Cân Bằng Nhãn

`compute_class_weights()` tính trọng số nghịch đảo với tần suất lớp, truyền vào `CrossEntropyLoss(weight=...)`. Class hiếm (Personality Disorder, ~900 mẫu) nhận weight cao hơn trong loss.

### Training Loop

1. Forward pass → compute weighted cross-entropy loss
2. Backward → Adam optimizer + gradient clipping (norm ≤ 5)
3. Evaluate trên val set sau mỗi epoch → tính macro F1
4. EarlyStopping theo macro F1 (patience=5) → lưu checkpoint tốt nhất
5. Dừng sau epoch 6/30 (early stopping triggered)

### Files Liên Quan

```
src/models/bilstm.py        → BiLSTMClassifier
src/training/trainer.py     → Training loop, EarlyStopping
src/data/dataset.py         → Vocabulary, SentimentDataset
scripts/train_bilstm.py     → Entry-point train
scripts/eval_bilstm.py      → Entry-point đánh giá
configs/bilstm.yaml         → Hyperparameters
```

---

## 7. Model 2 — BERTweet (M4)

### Đại Diện Cho: Kiến Trúc Transformer Pre-trained

BERTweet là mô hình ngôn ngữ dựa trên RoBERTa, được **pre-train đặc biệt trên 850 triệu tweet tiếng Anh** — lý tưởng cho social media text. Dự án sử dụng `vinai/bertweet-base` từ HuggingFace.

### Kiến Trúc

```
Input text
    ↓
AutoTokenizer (bertweet-base, max_len=128)
    → input_ids + attention_mask
    ↓
BERTweet Base (12 Transformer layers, 768-dim hidden)
    [Self-attention 2 chiều trên toàn bộ câu]
    ↓
[CLS] token representation (768-dim)
    ↓
Classification head: Linear(768 → 7) [fresh, khởi tạo mới]
    ↓
Logits (batch × 7)
```

### Điểm Khác Biệt So Với BiLSTM

| Khía cạnh       | BiLSTM              | BERTweet                     |
| --------------- | ------------------- | ---------------------------- |
| Context         | Sequential (LSTM)   | Full bidirectional attention |
| Pre-training    | Không (random init) | 850M tweets                  |
| Token           | Word-level vocab    | BPE subword tokenizer        |
| Tham số         | ~5M                 | ~135M                        |
| Checkpoint size | ~60 MB              | ~540 MB                      |

### Kỹ Thuật Huấn Luyện

- **Gradient Accumulation** (`steps=2`): Effective batch size = 16 × 2 = 32, giúp ổn định training trên GPU nhỏ
- **AdamW + Linear Warmup**: Learning rate tăng dần 6% đầu rồi giảm tuyến tính
- **Optional FP16**: Có thể bật `fp16: true` để tăng tốc trên GPU (mặc định off để an toàn trên CPU)
- **`freeze_base: false`**: Fine-tune toàn bộ encoder (không chỉ head) — cho kết quả tốt hơn
- **EarlyStopping**: Cùng metric `macro_f1`, patience=3 — đảm bảo so sánh công bằng với BiLSTM
- Dừng sau epoch 8

### Files Liên Quan

```
src/data/bertweet_dataset.py → TransformerSentimentDataset, build_transformer_loaders()
src/models/bertweet.py       → BERTweetClassifier (HuggingFace wrapper)
scripts/train_bertweet.py    → Entry-point train
scripts/eval_bertweet.py     → Entry-point đánh giá
configs/bertweet.yaml        → Hyperparameters
```

---

## 8. Model 3 — LLM Prompting (M5)

### Đại Diện Cho: Zero-shot / Few-shot với LLM

Thay vì huấn luyện từ đầu hoặc fine-tune, module này dùng **Gemini 2.5 Flash** (Google) để phân loại theo phương pháp **prompting** — không cập nhật trọng số mô hình.

### Hai Chế Độ Prompting

**Zero-shot** — Chỉ mô tả nhãn, không có ví dụ:

```
Classify the mental health sentiment of the following text.
Respond with JSON: {"label": "...", "confidence": 0.0–1.0, "explanation": "..."}

Labels: Normal, Depression, Anxiety, Bipolar, Personality disorder, Stress, Suicidal

Text: "I haven't slept in 3 days, my mind won't stop"
```

**Few-shot** — Cung cấp 1–2 ví dụ cho mỗi nhãn trước khi hỏi.

### Thiết Kế Kỹ Thuật

**Structured Output:** LLM được yêu cầu trả về JSON với 3 trường:

- `label`: nhãn phân loại
- `confidence`: độ tin cậy (0–1)
- `explanation`: lý giải tại sao

→ Parsing bằng `json.loads()` → fallback strip markdown fences → ~3% fail rate được ghi log.

**CostAccumulator:** Giới hạn ngân sách API (`budget_cap_usd=5.0`). Nếu chi phí ước tính vượt ngưỡng → raise `RuntimeError`, không để cost leo thang.

**JSONL Streaming:** Mỗi prediction được ghi vào file ngay sau khi nhận response — tránh mất dữ liệu khi crash giữa chừng.

**Sampling:** Chỉ lấy **200 mẫu** từ test set (sample_seed=42) cho dev run, giảm cost đáng kể. Full run có thể tăng lên.

### Chi Phí Thực Tế

| Model            | Chế độ    | Số mẫu | Chi phí ước tính |
| ---------------- | --------- | ------ | ---------------- |
| gemini-2.5-flash | zero-shot | 200    | ~$0.009 USD      |
| gemini-2.5-flash | few-shot  | 200    | ~$0.015 USD      |

### Security Considerations

API key chỉ đọc từ biến môi trường `GEMINI_API_KEY` — **không bao giờ hardcode** trong config hay source code. Text người dùng chỉ đưa vào _user turn_, không vào _system instruction_ — tránh prompt injection.

### Files Liên Quan

```
src/models/llm_client.py          → LLMClient, CostAccumulator, LLMPrediction
scripts/run_llm_prompting.py      → Entry-point batch inference
configs/llm_prompting.yaml        → Model, chế độ, budget, paths
data/artifacts/llm_predictions.jsonl   → Streaming output
data/artifacts/llm_fewshot_predictions.jsonl
```

---

## 9. Phân Tích Ngữ Nghĩa — Semantic Analysis (M6)

Module này là **điểm khác biệt quan trọng** của dự án — không chỉ phân loại mà còn phân tích ý nghĩa ở tầng sâu hơn.

### 9.1 Sentence Embeddings

**Model:** `all-MiniLM-L6-v2` từ `sentence-transformers`

- Đây là mô hình nhỏ gọn (80MB) nhưng hiệu quả cao cho semantic similarity
- Output: vector 384 chiều cho mỗi câu
- Embeddings được cache vào `data/artifacts/semantic_embeddings.npy` để tái sử dụng

### 9.2 Semantic Textual Similarity (STS)

Tính **cosine similarity** giữa các câu trong cùng class (within-class) và khác class (cross-class):

| Class                | Within-class cosine similarity |
| -------------------- | ------------------------------ |
| Depression           | **0.3304** ← cao nhất          |
| Bipolar              | 0.2891                         |
| Suicidal             | 0.2564                         |
| Anxiety              | 0.2394                         |
| Personality Disorder | 0.2030                         |
| Stress               | 0.1905                         |
| Normal               | 0.0917 ← thấp nhất             |
| **Cross-class avg**  | **0.1513**                     |

**Nhận xét:** Depression và Bipolar có STS trong lớp cao nhất — tức là chúng _có pattern ngôn ngữ đặc trưng rõ ràng_, giải thích tại sao các model phân loại tốt hơn. Normal có STS thấp vì nội dung language rất đa dạng.

Khoảng cách `within > cross` (0.26 average vs 0.15) cho thấy các nhãn có **ranh giới ngữ nghĩa có thể phân biệt được**.

### 9.3 Semantic Clustering (UMAP + HDBSCAN)

Quy trình giảm chiều và phân cụm:

```
Embeddings (N × 384)
    ↓
UMAP(n_components=2, metric=cosine, n_neighbors=30, random_state=42)
    ↓
2D projection (N × 2)
    ↓
HDBSCAN(min_cluster_size=120, min_samples=10)
    ↓
Cluster labels + noise points
```

**Mục đích:** Khám phá cấu trúc ẩn trong không gian embedding — liệu các nhãn có tạo thành cụm tách biệt không? Cluster nào bị overlap nhiều nhất?

**Kết quả:** Depression và Suicidal thường nằm gần nhau trong không gian embedding (overlap cao), phù hợp với thực tế lâm sàng rằng hai tình trạng này có nhiều triệu chứng chung.

### 9.4 Cross-Model Comparison

`run_comparison()` đọc cả 3 file metrics JSON và tạo bảng leaderboard so sánh, lưu vào `comparison_report.json`.

### Files Liên Quan

```
scripts/run_semantic_analysis.py   → Entry-point (CLI với --skip-sts, --skip-cluster)
data/artifacts/semantic_embeddings.npy   → Cache 384-dim embeddings
data/artifacts/semantic_embeddings_2d.npy → UMAP 2D projections
data/artifacts/sts_report.json           → STS metrics
data/artifacts/comparison_report.json   → Leaderboard 3 model
```

---

## 10. Kết Quả & So Sánh

### Bảng Kết Quả Tổng Hợp

| Model              | Accuracy   | Macro F1   | Weighted F1 | Ghi chú                     |
| ------------------ | ---------- | ---------- | ----------- | --------------------------- |
| **BERTweet**       | **0.8155** | **0.7831** | **0.8167**  | Fine-tuned 8 epochs, ~540MB |
| BiLSTM             | 0.7583     | 0.6936     | 0.7615      | Trained 6 epochs, ~60MB     |
| Gemini (zero-shot) | 0.6600     | 0.6175     | 0.6550      | 200 samples, $0.009         |
| Gemini (few-shot)  | 0.6150     | 0.5580     | —           | 200 samples                 |

### Kết Quả Chi Tiết Theo Class

#### BERTweet (tốt nhất tổng thể)

| Class                | Precision | Recall | F1    | Support |
| -------------------- | --------- | ------ | ----- | ------- |
| Normal               | 0.964     | 0.943  | 0.954 | 2,401   |
| Depression           | 0.805     | 0.694  | 0.745 | 2,264   |
| Anxiety              | 0.870     | 0.871  | 0.870 | 543     |
| Bipolar              | 0.811     | 0.845  | 0.828 | 375     |
| Personality Disorder | 0.591     | 0.702  | 0.642 | 134     |
| Stress               | 0.652     | 0.794  | 0.716 | 344     |
| Suicidal             | 0.679     | 0.784  | 0.728 | 1,596   |

#### BiLSTM (baseline)

| Class                | Precision | Recall | F1    | Support |
| -------------------- | --------- | ------ | ----- | ------- |
| Normal               | 0.934     | 0.903  | 0.918 | 2,401   |
| Depression           | 0.743     | 0.673  | 0.706 | 2,264   |
| Anxiety              | 0.780     | 0.764  | 0.772 | 543     |
| Bipolar              | 0.741     | 0.741  | 0.741 | 375     |
| Personality Disorder | 0.408     | 0.582  | 0.480 | 134     |
| Stress               | 0.501     | 0.651  | 0.566 | 344     |
| Suicidal             | 0.644     | 0.701  | 0.671 | 1,596   |

### Phân Tích Lỗi Quan Trọng

**1. Depression ↔ Suicidal confusion** (cả 3 model)

- BiLSTM nhầm 549/2264 Depression → Suicidal
- BERTweet nhầm 570/2264 Depression → Suicidal
- Giải thích: Hai class này overlap về mặt ngữ nghĩa — chứng minh bởi STS score cao giữa 2 class

**2. Personality Disorder — class khó nhất**

- BiLSTM F1 = 0.48, BERTweet F1 = 0.64
- Chỉ có 894 mẫu train (ít nhất trong tất cả class)
- Cụm text rất đa dạng, không có pattern ngôn ngữ rõ ràng

**3. LLM kém hơn kỳ vọng**

- Accuracy 0.66 thấp hơn dự báo ban đầu (88–92%)
- Nguyên nhân: Test chỉ trên 200 mẫu ≠ toàn bộ 7,657 mẫu; class imbalance ảnh hưởng đến few-shot examples; LLM không "hiểu" ranh giới label theo phân loại của dataset cụ thể này

**4. Few-shot kém hơn zero-shot**

- Few-shot accuracy 0.615 < zero-shot 0.660
- Hypothesis: Few-shot examples bị chọn từ minority class subset không đại diện tốt

### Insight Tổng Hợp

| Nhận xét                                     | Ý nghĩa                                                   |
| -------------------------------------------- | --------------------------------------------------------- |
| BERTweet vượt BiLSTM ~6% Macro F1            | Pre-training trên tweet domain có giá trị rõ ràng         |
| BiLSTM fail ở Personality Disorder (F1=0.48) | Class nhỏ + ngôn ngữ đa dạng → RNN không đủ capacity      |
| Depression STS within-class = 0.33 (cao)     | Depression có pattern ngôn ngữ nhất quán → dễ learn       |
| LLM zero-shot không phải "silver bullet"     | Không có fine-tuning → thiếu domain-specific boundaries   |
| Within-class STS > cross-class               | Nhãn có ranh giới ngữ nghĩa thực sự trong embedding space |

---

## 11. Cấu Trúc Dự Án

```
sentimind/
├── configs/                     # Tất cả hyperparameters
│   ├── preprocessing.yaml       # Cấu hình tiền xử lý
│   ├── bilstm.yaml              # BiLSTM config
│   ├── bertweet.yaml            # BERTweet config
│   ├── llm_prompting.yaml       # LLM config (model, budget, sample_size)
│   └── semantic.yaml            # Semantic analysis config
│
├── data/
│   ├── raw/                     # Dữ liệu gốc (không commit lên git)
│   │   └── Combined Data.csv    # Kaggle dataset (~31MB)
│   ├── processed/               # Dữ liệu đã xử lý
│   │   ├── train.csv            # 35,731 mẫu
│   │   ├── val.csv              # 7,657 mẫu
│   │   └── test.csv             # 7,657 mẫu
│   └── artifacts/               # Kết quả chạy
│       ├── vocab.json           # Word→Index mapping (BiLSTM)
│       ├── bilstm_best.pt       # Checkpoint BiLSTM (~60MB)
│       ├── bilstm_metrics.json  # Acc=0.758, Macro F1=0.694
│       ├── bertweet_best.pt     # Checkpoint BERTweet (~540MB)
│       ├── bertweet_metrics.json# Acc=0.816, Macro F1=0.783
│       ├── llm_metrics.json     # Acc=0.660, Macro F1=0.618 (zero-shot)
│       ├── llm_predictions.jsonl# Predictions + rationale
│       ├── semantic_embeddings.npy      # 384-dim embeddings
│       ├── semantic_embeddings_2d.npy   # UMAP 2D
│       ├── sts_report.json      # STS scores
│       ├── comparison_report.json       # Leaderboard 3 model
│       └── experiment_config.json       # Tóm tắt toàn bộ experiment
│
├── src/                         # Source code chính
│   ├── data/
│   │   ├── preprocess.py        # clean_text(), normalise_label()
│   │   ├── dataset.py           # Vocabulary, SentimentDataset, DataLoaders
│   │   └── bertweet_dataset.py  # TransformerSentimentDataset
│   ├── models/
│   │   ├── bilstm.py            # BiLSTMClassifier
│   │   ├── bertweet.py          # BERTweetClassifier (HuggingFace)
│   │   └── llm_client.py        # LLMClient, CostAccumulator, LLMPrediction
│   ├── training/
│   │   └── trainer.py           # Unified training loop + EarlyStopping
│   └── utils/
│       └── metrics.py           # compute_metrics(), confusion matrix
│
├── scripts/                     # Entry-point CLI
│   ├── preprocess.py            # Step 1
│   ├── train_bilstm.py          # Step 2a
│   ├── eval_bilstm.py           # Step 2b
│   ├── train_bertweet.py        # Step 3a
│   ├── eval_bertweet.py         # Step 3b
│   ├── run_llm_prompting.py     # Step 4
│   └── run_semantic_analysis.py # Step 5
│
├── tests/                       # Unit tests
│   ├── test_preprocessing.py    # 21 tests
│   ├── test_bilstm_model.py     # 20 tests
│   ├── test_bertweet_model.py   # 30 tests
│   ├── test_llm_client.py       # 19 tests
│   └── test_semantic_analysis.py# 18 tests
│
├── notebooks/                   # Jupyter notebooks (phân tích & báo cáo)
│   ├── m5_llm_gemini_prompting_eval.ipynb
│   ├── m6_semantic_analysis.ipynb
│   ├── m7_comprehensive_evaluation_report.ipynb
│   ├── m8_final_report.ipynb
│   └── rerun_full_pipeline.ipynb
│
├── docs/ai/                     # Tài liệu thiết kế
│   ├── design/                  # Feature design docs
│   ├── implementation/          # Implementation notes
│   └── PERSON_A_REVIEW.md       # Review notes
│
├── data_contract.md             # Hợp đồng schema dữ liệu (locked)
├── REPRODUCE.md                 # Hướng dẫn tái lập kết quả
├── init.md                      # Mô tả bài toán ban đầu
└── requirements.txt             # Python dependencies
```

---

## 12. Cách Chạy Lại Toàn Bộ Pipeline

### Yêu Cầu Hệ Thống

| Thành phần | Tối thiểu      | Khuyến nghị         |
| ---------- | -------------- | ------------------- |
| RAM        | 8 GB           | 16 GB               |
| GPU VRAM   | Không bắt buộc | 8 GB+ (NVIDIA CUDA) |
| Disk       | 2 GB           | 5 GB                |
| Python     | 3.10+          | 3.13                |

### Cài Đặt

```bash
# Clone và tạo môi trường
git clone <repo-url> sentimind
cd sentimind
conda create -n sentimind python=3.13 -y
conda activate sentimind
pip install -r requirements.txt

# Biến môi trường (chỉ cần cho bước LLM)
export GEMINI_API_KEY="your-api-key-here"
```

### Thứ Tự Chạy

```bash
# Bước 1: Tiền xử lý (M2) — ~2 phút
python scripts/preprocess.py

# Bước 2: Train BiLSTM (M3) — ~10-15 phút CPU / 3-5 phút GPU
python scripts/train_bilstm.py

# Bước 3: Đánh giá BiLSTM — ~1 phút
python scripts/eval_bilstm.py
# → Kỳ vọng: Accuracy ~0.758, Macro F1 ~0.694

# Bước 4: Train BERTweet (M4) — ~30-40 phút GPU / 3-4 giờ CPU
python scripts/train_bertweet.py

# Bước 5: Đánh giá BERTweet — ~5 phút
python scripts/eval_bertweet.py
# → Kỳ vọng: Accuracy ~0.816, Macro F1 ~0.783

# Bước 6: LLM Prompting (M5) — ~5-10 phút (~200 API calls)
# Yêu cầu: GEMINI_API_KEY đã set
python scripts/run_llm_prompting.py
# → Kỳ vọng: Accuracy ~0.660, Macro F1 ~0.618, Chi phí ~$0.009

# Bước 7: Semantic Analysis (M6) — ~3-5 phút (lần đầu)
python scripts/run_semantic_analysis.py
# → Sinh embeddings, STS report, clustering plot

# Bước 8: Chạy toàn bộ test suite
python -m pytest tests/ -v
# → Kỳ vọng: 108 tests passed
```

### Determinism Guarantees

| Thông số          | Giá trị  | Áp dụng                                                |
| ----------------- | -------- | ------------------------------------------------------ |
| Random seed       | 42       | Python, NumPy, PyTorch, data split, UMAP, LLM sampling |
| Stratified split  | 70/15/15 | train/val/test                                         |
| Min text length   | 3 ký tự  | Sau khi clean                                          |
| UMAP random_state | 42       | Semantic clustering                                    |
| LLM temperature   | 0.0      | Gemini inference                                       |

---

## 13. Điểm Mạnh, Hạn Chế & Hướng Mở Rộng

### Điểm Mạnh

| Khía cạnh                      | Mô tả                                                              |
| ------------------------------ | ------------------------------------------------------------------ |
| **Reproducibility cao**        | Seed cố định 42, deterministic pipeline, documented commands       |
| **Metric contract thống nhất** | 3 model cùng output format → so sánh công bằng                     |
| **Cost control**               | Budget cap ngăn runaway API cost                                   |
| **Streaming output**           | JSONL ghi từng prediction → không mất data khi crash               |
| **Test coverage tốt**          | 108 unit tests bao phủ toàn bộ modules                             |
| **Semantic layer**             | STS + clustering bổ sung chiều phân tích mà pure accuracy không có |
| **Cấu trúc sạch**              | Config-driven, không hardcode paths, modular scripts               |

### Hạn Chế

| Hạn chế                           | Chi tiết                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| **LLM chỉ test 200 mẫu**          | Kết quả không fully representative so với full test set      |
| **Few-shot kém zero-shot**        | Cần nghiên cứu thêm về few-shot example selection            |
| **Personality Disorder F1 thấp**  | Chỉ 894 mẫu — cần thêm data hoặc augmentation                |
| **Depression ↔ Suicidal overlap** | Ranh giới mờ về mặt ngôn ngữ và lâm sàng                     |
| **LLM không fine-tuned**          | Gemini không biết về label boundaries cụ thể của dataset này |
| **UMAP cần ~4GB RAM**             | Không phù hợp môi trường <4GB nếu dùng full test set         |

### Hướng Mở Rộng

| Hướng                          | Mô tả                                                                |
| ------------------------------ | -------------------------------------------------------------------- |
| **Fine-tune LLM**              | LoRA/QLoRA fine-tuning Gemma/LLaMA trên dataset → so sánh fair hơn   |
| **GloVe embeddings**           | Thử BiLSTM + GloVe Twitter pretrained embeddings                     |
| **Data augmentation**          | Tăng cường class Personality Disorder (back-translation, paraphrase) |
| **Multi-label classification** | Một post có thể thuộc nhiều class                                    |
| **Explainability**             | LIME/SHAP cho BiLSTM và BERTweet                                     |
| **LLM full run**               | Chạy LLM trên toàn bộ 7,657 test samples                             |
| **Ensemble**                   | Kết hợp prediction của 3 model                                       |
| **Real-time API**              | Xây dựng inference API endpoint                                      |

---

## Tóm Tắt Cho Trình Bày

### 3 Điểm Cần Nhớ

1. **BERTweet thắng tất cả** (Acc=0.816, F1=0.783) nhờ pre-training trên twitter domain — Transformer > RNN trong bài toán này
2. **LLM zero-shot không đạt tốt nhất** (Acc=0.66) — fine-tuning thực sự cần thiết, đặc biệt với domain-specific label definitions
3. **Semantic analysis xác nhận** ranh giới ngữ nghĩa giữa các class tồn tại (within > cross STS), và Depression↔Suicidal overlap giải thích confusion matrix

### Câu Trả Lời Cho Câu Hỏi Nghiên Cứu

> **"Transformer (BERTweet) hiểu ngữ nghĩa social media text tốt nhất trong task này, nhờ pre-training domain-specific. LLM mạnh về giải thích nhưng không có fine-tuning thì không tối ưu cho task có label boundaries đặc thù. RNN là baseline tốt và chi phí thấp nhất."**

---

_Tài liệu được tổng hợp từ toàn bộ codebase, design docs, và experiment artifacts của dự án Sentimind. Ngày cập nhật: 29/03/2026._
