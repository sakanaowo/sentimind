---
phase: planning
title: Main Task Board - Sentimind
description: Bang cong viec chinh cho toan bo de tai Mental Health Sentiment Analysis
---

# Main Task Board - Sentimind

## Tong quan pham vi

De tai: Mental Health Sentiment Analysis - so sanh BiLSTM (RNN) vs BERTweet (Transformer) vs LLM, ket hop semantic analysis (STS + clustering) va danh gia tong hop.

## Bang Cong Viec Chinh

| ID  | Cong viec chinh                            | Muc tieu                                            | Dau ra chinh                                                                         | Phu thuoc      | Uoc tinh   | Do uu tien | Trang thai |
| --- | ------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------- | ---------- | ---------- | ---------- |
| M1  | Khoi tao du an va chuan hoa tai lieu       | Thong nhat cau truc, quy uoc, command chay          | Cac file docs feature + quy uoc ten file + khung src/scripts                         | Khong          | 0.5 ngay   | Cao        | Done       |
| M2  | Thu thap va tien xu ly du lieu             | Tao input on dinh, tai lap duoc cho moi model       | Dataset da lam sach, label mapping, split train/val/test, bao cao chat luong du lieu | M1             | 1 ngay     | Cao        | Done       |
| M3  | Xay dung baseline BiLSTM                   | Co moc so sanh RNN ve chat luong va chi phi         | Script train/eval BiLSTM, checkpoint, metrics, confusion matrix                      | M2             | 1 ngay     | Cao        | Done       |
| M4  | Fine-tune BERTweet                         | Co baseline Transformer toi uu cho text mang xa hoi | Script fine-tune/eval, checkpoint tot nhat, metrics theo cung schema                 | M2             | 1-2 ngay   | Cao        | Todo       |
| M5  | LLM prompting va danh gia                  | Danh gia zero/few-shot + kha nang giai thich        | Prompt templates, inference script, file du doan + rationale + cost/latency          | M2             | 0.5-1 ngay | Trung binh | Todo       |
| M6  | Semantic analysis (STS + clustering)       | Do muc do hieu ngu nghia ngoai metrics classify     | STS report, embedding + UMAP/HDBSCAN clustering, hinh minh hoa                       | M3, M4, M5     | 1 ngay     | Cao        | Todo       |
| M7  | Danh gia tong hop va so sanh cong bang     | Tong hop ket qua 3 huong model theo 1 chuan         | Bang so sanh Accuracy/Precision/Recall/F1, error analysis, ket luan insight          | M3, M4, M5, M6 | 1 ngay     | Cao        | Todo       |
| M8  | Hoan thien bao cao va readiness trien khai | Chot tai lieu, rui ro, huong mo rong                | Tai lieu cuoi, checklist reproducibility, de xuat cai tien                           | M7             | 1 ngay     | Trung binh | Todo       |

## Thu tu trien khai de xuat

1. M1 -> M2
2. M3, M4, M5 (co the lam song song sau M2)
3. M6 (dua tren output cua M3-M5)
4. M7 -> M8

## Dinh nghia hoan thanh (Definition of Done) moi moc

- Co command chay ro rang va tai lap duoc.
- Co artifact dau ra luu trong thu muc quy uoc.
- Co cap nhat docs tuong ung trong requirements/design/planning/implementation/testing.
- Co ket qua kiem thu toi thieu (smoke test hoac evaluation check).

## Rui ro tong quan

- Mat can bang lop label gay lech metric: uu tien stratified split + class weighting.
- Gioi han compute khi fine-tune Transformer: dung gradient accumulation + early stopping.
- Chi phi API LLM: chay sampled benchmark truoc, dat budget tran.
- Clustering khong on dinh: co dinh seed va luu thong so UMAP/HDBSCAN.

## Phan cong cho 2 nguoi (doc lap toi da)

Nguyen tac tach viec:

- Chot hop dong du lieu ngay dau (schema input/output, ten cot, duong dan artifact, metric schema).
- Moi nhanh phat trien chi cham vao module rieng, tranh sua file chung khong can thiet.
- Chi dong bo o cac moc merge da quy dinh, khong doi den cuoi moi gop.

| Nguoi                                  | Cum cong viec chinh              | Nhiem vu cu the                                                                          | Dau ra ban giao                                                          | Muc do phu thuoc                     |
| -------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------ |
| Nguoi A (Data + RNN)                   | Foundation va baseline RNN       | M1 (chon quy uoc), M2 (preprocessing), M3 (BiLSTM), test cho pipeline du lieu            | Data contract, split artifacts, script train/eval BiLSTM, metrics BiLSTM | Doc lap sau khi chot contract        |
| Nguoi B (Transformer + LLM + Semantic) | Huong model nang cao va semantic | M4 (BERTweet), M5 (LLM prompting), M6 (STS + clustering), draft M7 phan semantic insight | Script BERTweet/LLM, ket qua semantic, bao cao chi phi va explainability | Doc lap sau khi nhan split artifacts |

Phan viec giao nhau (nho, co lich co dinh):

- M1 ngay dau: chot data contract va metric contract (2 nguoi cung review).
- M7: hop nhat bang so sanh cuoi cung, doi chieu metric schema va viet ket luan.
- M8: tong hop tai lieu va chot checklist reproducibility.

## Git Workflow + CI/CD de lam song song an toan

### Nhanh va quy tac merge

- Nhanh chinh: `main` (bao ve, khong push truc tiep).
- Nhanh tich hop theo nguoi:
  - `dev/person-a`
  - `dev/person-b`
- Nhanh tinh nang:
  - `feat/person-a-<task>`
  - `feat/person-b-<task>`
- Luong merge:
  1.  `feat/*` -> PR vao `dev/person-*` (bat buoc CI xanh)
  2.  Cuoi ngay hoac het moc: `dev/person-*` -> PR vao `main`
  3.  Bat buoc 1 reviewer la nguoi con lai

### CI toi thieu cho moi PR

- Job 1: Kiem tra dinh dang va lint (Python + Markdown).
- Job 2: Smoke test pipeline du lieu (tap mau nho).
- Job 3: Smoke eval (kiem tra schema metrics/outputs dung contract).
- Job 4: Kiem tra docs planning/requirements duoc cap nhat khi co thay doi lon.

### Quy uoc tranh xung dot

- Khong sua truc tiep file contract da chot neu chua thong nhat.
- Neu can thay contract: tao PR rieng `contract-change/*`, merge xong moi tiep tuc code.
- Rebase hoac merge `main` vao nhanh ca nhan it nhat 1 lan/ngay.

### DoD cho PR (ap dung ca 2 nguoi)

- CI pass 100%.
- Co cap nhat docs lien quan trong `docs/ai/*`.
- Co artifact mau de nguoi kia tai lap.
- Co mo ta ro input/output va cach chay.
