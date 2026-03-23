# Huong Dan Tai Lap Ket Qua (Reproducibility Guide)

> Tai lieu nay mo ta day du cac buoc de tai lap toan bo ket qua cua du an Sentimind
> tu du lieu tho den bao cao danh gia cuoi cung.

---

## 1. Yeu Cau He Thong

### Phan cung

| Thanh phan | Toi thieu | Khuyen nghi |
|------------|-----------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | Khong bat buoc (CPU fallback) | 8 GB+ (NVIDIA CUDA) |
| Disk | 2 GB trong | 5 GB trong |

### Phan mem

| Thanh phan | Phien ban da kiem chung |
|------------|------------------------|
| OS | Linux x86_64 (Ubuntu) |
| Python | 3.13.11 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.3.0 |
| sentence-transformers | 5.2.3 |
| scikit-learn | 1.8.0 |
| umap-learn | 0.5.11 |
| hdbscan | 0.8.41 |
| numpy | 2.4.2 |
| pandas | 3.0.0 |
| google-genai | >=1.0.0 |

---

## 2. Cai Dat Moi Truong

```bash
# Clone repository
git clone <repo-url> sentimind
cd sentimind

# Tao moi truong (conda hoac venv)
conda create -n sentimind python=3.13 -y
conda activate sentimind

# Cai dat dependencies
pip install -r requirements.txt
```

### Bien moi truong can thiet

| Bien | Muc dich | Bat buoc |
|------|----------|----------|
| GEMINI_API_KEY | Truy cap Gemini API cho M5 | Chi khi chay LLM prompting |

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## 3. Du Lieu Dau Vao

### Dataset chinh

- **Nguon**: Kaggle Mental Health Sentiment Dataset
- **Link**: `kaggle.com/datasets/suchintikasarkat/sentiment-analysis-for-mental-health`
- **Dat tai**: `data/raw/Combined Data.csv`

Kich thuoc: ~31 MB, 53,043 dong goc.

### Kiem tra du lieu da co

```bash
ls -la data/raw/
# Ky vong: Combined Data.csv (~31 MB)
```

---

## 4. Thu Tu Chay Pipeline

### Buoc 1: Tien xu ly du lieu (M2)

```bash
python scripts/preprocess.py
```

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/processed/train.csv` | 35,731 mau (70%) |
| `data/processed/val.csv` | 7,657 mau (15%) |
| `data/processed/test.csv` | 7,657 mau (15%) |
| `data/artifacts/preprocessing_report.json` | Bao cao chat luong |
| `data/artifacts/vocab.json` | Tu dien (word -> index) |

**Kiem tra:** 51,045 mau sau khi drop null/short/duplicate tu 53,043 goc.

### Buoc 2: Train BiLSTM baseline (M3)

```bash
python scripts/train_bilstm.py
```

**Thoi gian uoc tinh:** ~10-15 phut (CPU), ~3-5 phut (GPU)

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/bilstm_best.pt` | Checkpoint tot nhat (~60 MB) |
| `data/artifacts/bilstm_train_history.json` | Lich su train 19 epochs |

### Buoc 3: Danh gia BiLSTM (M3)

```bash
python scripts/eval_bilstm.py
```

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/bilstm_metrics.json` | Metrics theo data contract |
| `data/artifacts/bilstm_confusion_matrix.png` | Ma tran nham lan |

**Ket qua ky vong:** Accuracy ~0.758, Macro F1 ~0.694

### Buoc 4: Train BERTweet (M4)

```bash
python scripts/train_bertweet.py
```

**Thoi gian uoc tinh:** ~30-40 phut (GPU 8GB), ~3-4 gio (CPU)

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/bertweet_best.pt` | Checkpoint tot nhat (~540 MB) |
| `data/artifacts/bertweet_train_history.json` | Lich su train |

### Buoc 5: Danh gia BERTweet (M4)

```bash
python scripts/eval_bertweet.py
```

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/bertweet_metrics.json` | Metrics theo data contract |
| `data/artifacts/bertweet_confusion_matrix.png` | Ma tran nham lan |

**Ket qua ky vong:** Accuracy ~0.816, Macro F1 ~0.783

### Buoc 6: LLM Prompting (M5)

```bash
# Yeu cau: GEMINI_API_KEY da set
python scripts/run_llm_prompting.py
```

**Thoi gian uoc tinh:** ~5-10 phut (200 API calls)
**Chi phi uoc tinh:** ~$0.009 USD (zero-shot, gemini-2.5-flash)

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/llm_predictions.jsonl` | Du doan + rationale |
| `data/artifacts/llm_metrics.json` | Metrics zero-shot |
| `data/artifacts/llm_confusion_matrix.png` | Ma tran nham lan |
| `data/artifacts/llm_cost_report.json` | Bao cao chi phi |

**Ket qua ky vong:** Accuracy ~0.66, Macro F1 ~0.618

### Buoc 7: Semantic Analysis (M6)

```bash
python scripts/run_semantic_analysis.py
```

**Thoi gian uoc tinh:** ~3-5 phut (tinh embeddings lan dau), <1 phut (dung cache)

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/semantic_embeddings.npy` | Embeddings 384-dim |
| `data/artifacts/semantic_embeddings_2d.npy` | UMAP 2D projections |
| `data/artifacts/sts_report.json` | STS within/cross class |
| `data/artifacts/semantic_cluster_plot.png` | HDBSCAN cluster plot |
| `data/artifacts/comparison_report.json` | So sanh 3 model |

### Buoc 8: Danh gia tong hop (M7)

```bash
jupyter notebook notebooks/m7_comprehensive_evaluation_report.ipynb
# Chay tat ca cells tu tren xuong
```

**Dau ra:**

| File | Mo ta |
|------|-------|
| `data/artifacts/model_comparison_bar.png` | Bieu do so sanh |
| `data/artifacts/confusion_matrices_combined.png` | Ma tran nham lan 3 model |

---

## 5. Chay Tests

```bash
# Chay toan bo test suite
python -m pytest tests/ -v

# Ky vong: 108 passed
```

Danh sach test modules:

| File | So test | Pham vi |
|------|---------|---------|
| `test_preprocessing.py` | 21 | Text cleaning, label mapping, split |
| `test_bilstm_model.py` | 20 | BiLSTM forward, trainer, checkpoint |
| `test_bertweet_model.py` | 30 | BERTweet forward, tokenizer, checkpoint |
| `test_llm_client.py` | 19 | LLM client, prompt generation, parsing |
| `test_semantic_analysis.py` | 18 | STS, clustering, embeddings, comparison |

---

## 6. Cau Truc Artifact Cuoi Cung

```
data/artifacts/
  -- Preprocessing --
  preprocessing_report.json     Bao cao tien xu ly
  vocab.json                    Tu dien BiLSTM

  -- BiLSTM (M3) --
  bilstm_best.pt                Checkpoint (~60 MB)
  bilstm_metrics.json           Acc=0.758, F1=0.694
  bilstm_confusion_matrix.png   Ma tran nham lan
  bilstm_train_history.json     19 epochs

  -- BERTweet (M4) --
  bertweet_best.pt              Checkpoint (~540 MB)
  bertweet_metrics.json         Acc=0.816, F1=0.783
  bertweet_confusion_matrix.png Ma tran nham lan
  bertweet_train_history.json   8 epochs
  bertweet_train_curves.png     Learning curves

  -- LLM Gemini (M5) --
  llm_predictions.jsonl         Du doan zero-shot (200 mau)
  llm_metrics.json              Acc=0.660, F1=0.618
  llm_confusion_matrix.png      Ma tran nham lan
  llm_cost_report.json          Chi phi API
  llm_fewshot_predictions.jsonl Du doan few-shot
  llm_fewshot_metrics.json      Acc=0.615, F1=0.558
  llm_fewshot_confusion_matrix.png

  -- Semantic (M6) --
  semantic_embeddings.npy       Embeddings goc (384-dim)
  semantic_embeddings_2d.npy    UMAP 2D
  sts_report.json               STS scores
  semantic_cluster_plot.png     HDBSCAN clustering
  comparison_report.json        So sanh tong hop 3 model

  -- Evaluation (M7) --
  model_comparison_bar.png      Bieu do so sanh
  confusion_matrices_combined.png  Ma tran 3 model
```

---

## 7. Thong So Co Dinh (Determinism)

| Thong so | Gia tri | Ap dung |
|----------|---------|---------|
| Random seed | 42 | Python, NumPy, PyTorch, split, sampling |
| Stratified split | 70/15/15 | train/val/test |
| Min text length | 3 ky tu | Sau khi clean |
| UMAP random_state | 42 | Semantic clustering |
| LLM temperature | 0.0 | Gemini inference |
| LLM sample_seed | 42 | Chon 200 mau tu test set |

---

## 8. Cau Hinh (Configs)

Toan bo hyperparameters duoc quan ly tap trung trong `configs/`:

| File | Model / Module |
|------|----------------|
| `configs/preprocessing.yaml` | Pipeline tien xu ly |
| `configs/bilstm.yaml` | BiLSTM (embedding, hidden, dropout, lr, epochs) |
| `configs/bertweet.yaml` | BERTweet (pretrained, batch, lr, warmup, fp16) |
| `configs/llm_prompting.yaml` | Gemini (model, temperature, max_tokens, sample) |
| `configs/semantic.yaml` | STS + UMAP + HDBSCAN + comparison paths |

---

## 9. Xu Ly Su Co Thuong Gap

| Van de | Nguyen nhan | Giai phap |
|--------|-------------|-----------|
| CUDA out of memory (BERTweet) | Batch size qua lon | Giam `batch_size` trong `configs/bertweet.yaml` |
| LLM parse errors | Thinking tokens tieu budget | Da fix: `ThinkingConfig(thinking_budget=0)` |
| Chi 2 clusters (HDBSCAN) | Params mac dinh qua tho | Da tune: min_cluster_size=120, n_neighbors=30 |
| Module thay doi khong co hieu luc | Notebook cache | Dung `importlib.reload()` |
| Few-shot kem hon zero-shot | Gemini 2.5 flash behavior | Ghi nhan; zero-shot la ket qua chinh |

---

## 10. Metric Contract

Tat ca model xuat metrics theo cung schema (xem chi tiet trong `data_contract.md` muc 7):

```json
{
  "model": "<string>",
  "split": "test",
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "weighted_f1": 0.0,
  "per_class": {
    "<label>": { "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0 }
  },
  "confusion_matrix": [[0]]
}
```
