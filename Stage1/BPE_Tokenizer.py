import tiktoken

class BPETokenizer:
    """Reusable BPE tokenizer wrapper using tiktoken."""

    def __init__(self, encoding_name="gpt2", allowed_special=None):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.allowed_special = set(allowed_special or [])
        self.vocab_size = self.tokenizer.max_token_value + 1

    def encode(self, text, allowed_special=None):
        allowed = allowed_special if allowed_special is not None else self.allowed_special
        return self.tokenizer.encode(text, allowed_special=allowed)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def encode_batch(self, texts, allowed_special=None):
        allowed = allowed_special if allowed_special is not None else self.allowed_special
        return [self.tokenizer.encode(text, allowed_special=allowed) for text in texts]
            


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    data_path = r"D:\AI-Projects\LLM-From-Scratch\HarryPotterDataset\all_harry_potter.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text)
    print("Encoded IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))