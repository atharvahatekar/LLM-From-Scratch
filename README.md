# LLM From Scratch

A ground-up implementation of a GPT-style Large Language Model trained on the Harry Potter book series. Each stage is built independently — no high-level ML frameworks used for the core model logic, only PyTorch primitives and NumPy.

> **Status:** Stage 1 complete — tokenization pipeline and data loading. Stages 2–4 in progress.

---

## Project Roadmap

| Stage | Topic | Status |
|-------|-------|--------|
| 1 | Tokenization, Embeddings & Data Pipeline | ✅ Complete |
| 2 | Attention Mechanism (Scaled Dot-Product, Multi-Head) | 🔄 In Progress |
| 3 | GPT Architecture (Transformer Blocks, Layer Norm, Feed-Forward) | 🔜 Planned |
| 4 | Pre-training & Text Generation | 🔜 Planned |

---

## Stage 1 — Tokenization & Data Pipeline

### `stage1_tokenization/`

| File | Description |
|------|-------------|
| `Tokenizer_V1.py` | Rule-based tokenizer using regex splits. Builds a vocabulary from raw text and supports encode/decode. |
| `Tokenizer_V2.py` | Extended V1 with `<\|unk\|>` and `<\|endoftext\|>` special token support for out-of-vocabulary handling. |
| `BPE_Tokenizer.py` | Byte Pair Encoding tokenizer wrapping OpenAI's `tiktoken` (GPT-2 encoding). Supports batch encode/decode. |
| `dataset_loader.py` | PyTorch `Dataset` + `DataLoader` using a sliding window strategy to create input/target sequence pairs. |
| `token_embeddings.py` | Token embedding layer mapping token IDs to dense vectors (in progress). |

### How the data pipeline works

```
Raw text  →  BPETokenizer.encode()  →  token IDs
                                           ↓
                              GPTDataset (sliding window)
                           [tok_0, tok_1, ... tok_N]   ← input
                           [tok_1, tok_2, ... tok_N+1] ← target
                                           ↓
                              DataLoader (batched)
```

The sliding window creates overlapping context windows with configurable `max_length` and `stride`, matching the pre-training data format used in GPT-2.

---

## Notebooks

Step-by-step Jupyter notebooks documenting the learning process behind each component, with explanations, visualizations, and experiments.

```
notebooks/stage1_tokenization/
├── 01_tokenizer.ipynb           # Rule-based tokenization from scratch
├── 02_bpe_tokenizer.ipynb       # Byte Pair Encoding deep dive
├── 03_data_loader.ipynb         # Sliding window dataset construction
├── 04_vector_embeddings.ipynb   # Dense vector representations
├── 05_token_embeddings.ipynb    # Token embedding layer
├── 06_position_embeddings.ipynb # Positional encoding
├── 07_attention_mechanism.ipynb # Self-attention from scratch
└── 08_llm_architecture.ipynb    # Full GPT block architecture
```

---

## Setup

```bash
git clone https://github.com/your-username/LLM-From-Scratch.git
cd LLM-From-Scratch
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Dataset

The model trains on the Harry Potter book series. The dataset files are not included in this repository due to size and copyright. Place your text files in a `data/` directory — see [data/README.md](data/README.md) for details.

---

## Quick Start

```python
from stage1_tokenization.BPE_Tokenizer import BPETokenizer
from stage1_tokenization.dataset_loader import create_dataloader

with open("data/all_harry_potter.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
dataloader = create_dataloader(text, batch_size=4, max_length=256, stride=128)

for input_ids, target_ids in dataloader:
    print("Input shape:", input_ids.shape)   # [4, 256]
    print("Target shape:", target_ids.shape) # [4, 256]
    break
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Tensor ops, Dataset/DataLoader, model layers |
| `tiktoken` | GPT-2 BPE tokenizer (OpenAI) |
| `regex` | Advanced pattern matching for rule-based tokenizer |
| `numpy` | Numerical utilities |
| `pandas` / `pyarrow` | Dataset inspection and Parquet support |
