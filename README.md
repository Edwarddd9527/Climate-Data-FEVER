# Evidence Retrieval & Claim Verification (BM25+NER → BiLSTM+SVM)

A modular pipeline for **evidence retrieval** and **claim verification**.  
It combines **BM25 with NER-informed query expansion** for retrieval, and a **two-stage classifier** (Doc2Vec + BiLSTM, followed by SVM) for final label prediction.

![Pipeline](docs/approach.png "Approach Overview — replace with your figure")

## Key Features
- **Retrieval-first design**: BM25 with **NER duplication/expansion** and **synonym augmentation** (WordNet). Tuned parameters: `k1=0.6`, `b=0.6`.
- **Two-stage classification**:
  - **Stage 1** — *One-to-one*: Doc2Vec embeddings → **BiLSTM** → class probabilities per (claim, evidence).
  - **Stage 2** — *One-to-many*: **SVM** fusing top-k evidences’ signals into a final decision.
- **Low-resource friendly**: synonym-based augmentation, simple top-*n* selection (default `n=4`), optional transformer encoder for re-ranking.

---

## Tasks & Data
- **Input**: claims (`train/dev/test` in JSONL) and a large evidence corpus (JSONL, ~1M+ snippets).
- **Labels**: `SUPPORT`, `REFUTE`, `NOT_ENOUGH_INFO` (NEI). The `DISPUTED` label can be omitted in training to reduce ambiguity.
- **Preprocessing**:
  - Clean non-English tokens/symbols.
  - **NER (spaCy)** to detect entities. Two strategies:
    - **duplicate** — inject replicated entities into the token stream.
    - **expand** — type-based weights; keep **two context tokens** before/after entities.
  - Lemmatization & stopword removal.


---

## Method

### Evidence Retrieval
1. **Synonym expansion** — create 3 variants per claim with **NLTK WordNet**.
2. **Weighted BM25** — score the original claim (weight **1.5**) and variants (weight **1.0** each); sum per-evidence scores.
3. **Top-*n*** — select top `n=4` evidences (tuneable).  
4. *(Optional)* **Transformer re-ranking** — pretrain a masked-LM on the evidence corpus, then re-rank BM25 results with a binary classifier.

![BM25 Heatmap](docs/bm25_heatmap.png "BM25 parameter heatmap — replace with your own plot")

### Classification
- **Stage 1: Doc2Vec + BiLSTM**
  - Doc2Vec (gensim), `vector_size=200`, `epochs=50`.
  - BiLSTM: `embed_dim=200`, `hidden=256`, **Adam**, cross-entropy, early stopping (≤400 epochs).
  - Output: 3-d probability per (claim, evidence).
- **Stage 2: SVM fusion**
  - For each claim, aggregate signals across **top-50** evidences (incl. Stage-1 probabilities) to output the final label.
- **Augmentation**
  - Synonym replacement to mitigate class imbalance (more for `SUPPORT/REFUTE`, fewer for `NEI/DISPUTED`).

---

## Results (Example)
A representative run produced the following accuracies:

| Method                                                          | Accuracy |
| --------------------------------------------------------------- | :------: |
| Transformer encoder                                             |  0.31    |
| Doc2Vec + BiLSTM + Logistic Regression                          |  0.38    |
| **Synonym** + Doc2Vec + BiLSTM + Logistic Regression           |  0.42    |
| **Synonym + Doc2Vec + BiLSTM + SVM (final)**                    | **0.46** |
| **Synonym + Transformer encoder**                                |  0.43    |

> Notes: synonym augmentation improved BiLSTM by **~+9.5%** (Doc2Vec randomness applies) and improved a transformer encoder by **~+27.9%**. Tuning top-*n* and BM25 parameters can shift these numbers.

## License
**MIT License** — see `LICENSE`.

---

## References
- Diggelmann et al., 2021. *Climate-FEVER: A Dataset for Verification of Real-World Climate Claims.* arXiv:2012.00614.  
- Lau & Baldwin, 2016. *An Empirical Evaluation of Doc2Vec…* arXiv:1607.05368.  
- Robertson & Zaragoza, 2009. *BM25 and Beyond.* Foundations and Trends in IR.  
- Soleimani et al., 2019. *BERT for Evidence Retrieval and Claim Verification.* arXiv:1910.02655.  
- Thorne et al., 2018. *FEVER: A Large-Scale Dataset for Fact Extraction and Verification.* arXiv:1803.05355.
