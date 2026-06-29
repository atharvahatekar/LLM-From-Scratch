# Dataset

The training corpus used in this project is the Harry Potter book series (Books 1–7).

Dataset files are **not included** in this repository. To set up the data directory:

## Option 1 — Hugging Face dataset

```python
from datasets import load_dataset

ds = load_dataset("manu/project_gutenberg", split="train")
```

Or use the Harry Potter specific dataset:

```
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
```

## Option 2 — Raw text files

Place plain `.txt` files in this `data/` directory:

```
data/
├── 1-Harry-Potter-and-the-Sorcerers-Stone.txt
├── 2-Harry-Potter-and-the-Chamber-of-Secrets.txt
├── ...
└── all_harry_potter.txt   ← concatenated corpus (used for training)
```

To concatenate all books into a single file:

```python
import os

books = sorted([f for f in os.listdir("data/") if f.endswith(".txt") and f != "all_harry_potter.txt"])
with open("data/all_harry_potter.txt", "w", encoding="utf-8") as out:
    for book in books:
        with open(f"data/{book}", "r", encoding="utf-8") as f:
            out.write(f.read())
            out.write("\n\n")
```

The `data/` directory is gitignored — no large files are committed to this repo.
