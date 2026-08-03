import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        self.vocab_size = 4 
        text = " ".join(texts)
        texts = sorted(text.split(" "),reverse=False)
        for i,t in enumerate(texts):
            if not self.word_to_id.get(t.lower(),None):
                self.word_to_id[t.lower()] = self.vocab_size
                self.vocab_size += 1

        self.id_to_word = {id:t for t,id in self.word_to_id.items()}
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = []
        for t in text.strip().split():
            ids.append(self.word_to_id.get(t.lower(),1))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        text = []
        for id in ids:
            token = self.id_to_word.get(id) or self.id_to_word.get(1)
            text.append(token)
        return " ".join(text)