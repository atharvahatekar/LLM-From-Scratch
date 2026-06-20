import re

class SimpleTokenizerV1:
    """
    A minimal rule-based tokenizer that
    splits text on whitespace and common punctuation, builds or accepts a token->id
    vocabulary, and supports `encode` / `decode` operations for text <-> id lists.
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
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

if __name__ == "__main__":
    
    text = """"It's the last he painted, you know," 
            Mrs. Gisburn said with pardonable pride."""
    
    tokenizer = SimpleTokenizerV1(text)


    ids = tokenizer.encode(text)
    print(ids)
    
    decoded_text = tokenizer.decode(ids)
    print(decoded_text)