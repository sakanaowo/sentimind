# Chi tiết Đề tài #1

## 🎯 "Mental Health Sentiment Analysis: Comparing RNN vs Transformer vs LLM"

---

## 1. Tổng quan bài toán

**Bối cảnh:**
Mạng xã hội (Twitter/Reddit) là nơi người dùng chia sẻ cảm xúc liên quan đến sức khỏe tâm thần. Bài toán đặt ra là: **Làm sao máy tính hiểu được cảm xúc phức tạp trong các đoạn text này?**

**Câu hỏi nghiên cứu chính:**

> _"Trong bài toán phân loại cảm xúc sức khỏe tâm thần, model nào (RNN / Transformer / LLM) hiểu ngữ nghĩa (semantic) tốt hơn và tại sao?"_

---

## 2. Datasets sử dụng

### Dataset 1 — Kaggle Mental Health

```
Link: kaggle.com/datasets/suchintikasarkat/sentiment-analysis-for-mental-health
```

| Cột     | Mô tả                                             |
| ------- | ------------------------------------------------- |
| `text`  | Câu/đoạn văn từ social media                      |
| `label` | Normal / Depression / Anxiety / Bipolar / PTSD... |

**Ví dụ dữ liệu:**

| text                                             | label      |
| ------------------------------------------------ | ---------- |
| "I can't get out of bed, nothing feels worth it" | Depression |
| "My heart races every time I leave the house"    | Anxiety    |
| "Today was actually a good day, felt productive" | Normal     |

---

### Dataset 2 — TweetEval (Sentiment subset)

```
Link: huggingface.co/datasets/cardiffnlp/tweet_eval
Subset: sentiment (negative / neutral / positive)
```

Dùng để **pre-train / fine-tune** model trên dữ liệu tweet trước, rồi transfer sang mental health.

---

## 3. Phần Semantic — Đây là điểm khác biệt quan trọng

Không chỉ classify đơn thuần, bài toán cần thêm **Semantic Analysis** để phân tích ý nghĩa sâu hơn:

### 3.1 Semantic Textual Similarity (STS)

So sánh độ tương đồng ngữ nghĩa giữa các câu:

```
"I feel hopeless and empty"
"There's no point in anything anymore"
→ Semantic similarity: 0.91 (rất giống nhau → cùng Depression)
```

### 3.2 Topic Modeling bằng Semantic Clustering

Dùng **sentence-transformers** để embed câu → cluster → tìm ra các chủ đề ẩn:

```
Cluster 1: Loneliness / Isolation
Cluster 2: Anxiety / Panic
Cluster 3: Hopelessness / Suicidal ideation
Cluster 4: Recovery / Positive coping
```

### 3.3 Semantic Role trong LLM

Dùng LLM (GPT/LLaMA) để giải thích **tại sao** một câu được classify là Depression, không chỉ đưa ra label.

---

## 4. Ba Models cần implement

### Model 1 — BiLSTM (đại diện RNN)

```python
# Kiến trúc
Embedding → BiLSTM → Dropout → Dense → Softmax

# Semantic: dùng pre-trained word2vec/GloVe embeddings
```

**Ưu điểm:** Nhanh, nhẹ, baseline tốt
**Nhược điểm:** Không hiểu context dài, không nắm được semantic sâu

---

### Model 2 — BERTweet (đại diện Transformer)

```python
# Model có sẵn, chỉ cần fine-tune
from transformers import AutoModel
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
)
```

**Ưu điểm:** Hiểu context 2 chiều, pre-trained trên Twitter data
**Nhược điểm:** Cần GPU, chậm hơn RNN

---

### Model 3 — LLM (GPT-4o-mini hoặc LLaMA-3)

```python
# Zero-shot hoặc Few-shot prompting
prompt = """
Classify the mental health sentiment of this text.
Labels: Normal, Depression, Anxiety, Bipolar, PTSD

Text: "I haven't slept in 3 days, my mind won't stop"
Answer:
"""
```

**Ưu điểm:** Không cần train, hiểu ngữ nghĩa rất tốt, có thể giải thích
**Nhược điểm:** Tốn API cost, không kiểm soát được hoàn toàn

---

## 5. Pipeline tổng thể

```
┌─────────────────────────────────────────┐
│           RAW DATA                      │
│  Kaggle Mental Health + TweetEval       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         PREPROCESSING                   │
│  - Remove URLs, @mentions, #hashtags    │
│  - Lowercase, tokenize                  │
│  - Handle class imbalance (SMOTE)       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│       SEMANTIC ANALYSIS                 │
│  - STS scoring (sentence-transformers)  │
│  - Semantic clustering (UMAP + HDBSCAN) │
│  - Visualize với t-SNE                  │
└────────────────┬────────────────────────┘
                 ↓
       ┌─────────┴──────────┐
       ↓         ↓          ↓
   BiLSTM    BERTweet    LLM (GPT)
  (RNN)   (Transformer) (Few-shot)
       ↓         ↓          ↓
       └─────────┬──────────┘
                 ↓
┌─────────────────────────────────────────┐
│         EVALUATION & SO SÁNH            │
│  - Accuracy, F1-score, Precision/Recall │
│  - Semantic similarity score            │
│  - Confusion matrix                     │
│  - Error analysis                       │
└─────────────────────────────────────────┘
```

---

## 6. Kết quả kỳ vọng

| Model     | Accuracy (dự kiến) | Semantic Understanding | Chi phí    |
| --------- | ------------------ | ---------------------- | ---------- |
| BiLSTM    | ~75-80%            | Thấp                   | Rất thấp   |
| BERTweet  | ~85-90%            | Cao                    | Trung bình |
| LLM (GPT) | ~88-92%            | Rất cao                | Cao        |

**Insight thú vị có thể rút ra:**

- LLM giỏi zero-shot nhưng BERTweet fine-tuned có thể không kém hơn nhiều
- RNN fail ở những câu có **sarcasm** hoặc **negation** ("I'm _fine_" — thực ra không fine)
- Semantic clustering giúp phát hiện pattern mà label gốc bỏ sót

---

## 7. Độ phức tạp thực tế

| Phần                      | Thời gian ước tính |
| ------------------------- | ------------------ |
| Data preprocessing        | 1 ngày             |
| BiLSTM implementation     | 1 ngày             |
| BERTweet fine-tuning      | 1-2 ngày           |
| LLM prompting             | vài giờ            |
| Semantic analysis         | 1 ngày             |
| Evaluation + viết báo cáo | 2 ngày             |
| **Tổng**                  | **~1 tuần**        |

---

## 8. Tài nguyên tham khảo có sẵn

```
📦 Code baseline TweetEval:
   github.com/cardiffnlp/tweeteval

🤗 Model BERTweet:
   huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

📓 Notebook Kaggle mẫu:
   Tìm "mental health sentiment BERT" trên kaggle.com/code

📚 Thư viện semantic:
   pip install sentence-transformers  ← STS + clustering
   pip install umap-learn hdbscan     ← visualization
```
