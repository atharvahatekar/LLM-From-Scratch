import re

class SimpleTokenizerV2:
    """
    Rule-based tokenizer mapping tokens <-> integer ids.
    Splits on whitespace and common punctuation and unknown tokens map to "<|unk|>".
    """
    def __init__(self, vocab_or_text):
        if isinstance(vocab_or_text, str):
            tokens = [t.strip() for t in re.split(r'([,.:;?_!"()\']|--|\s)', vocab_or_text) if t.strip()]
            self.str_to_int = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
        else:
            self.str_to_int = vocab_or_text
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    

if __name__ == "__main__":
    
    text = """"It's the last he painted, you know," 
            Mrs. Gisburn said with pardonable pride."""

    tokens = [t.strip() for t in re.split(r'([,.:;?_!"()\']|--|\s)', text) if t.strip()]

    vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}

    # remove 'pardonable' so it becomes the only unknown
    vocab.pop('pardonable', None)

    if "<|unk|>" not in vocab:
        vocab["<|unk|>"] = max(vocab.values(), default=-1) + 1

    tokenizer = SimpleTokenizerV2(vocab)

    print("tokens:", tokens)
    ids = tokenizer.encode(text)
    print("ids:", ids)
    print("decoded:", tokenizer.decode(ids))