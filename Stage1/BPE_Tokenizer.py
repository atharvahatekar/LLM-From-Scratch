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
    tokenizer = BPETokenizer("gpt2", allowed_special={"<|endoftext|>"})
    text = """It's the last he painted, you know," 
            Mrs. Gisburn said with pardonable pride.<|endoftext|>"""
    ids = tokenizer.encode(text)
    print("Encoded IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))