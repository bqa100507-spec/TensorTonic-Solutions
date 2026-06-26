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
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for index, token in enumerate(special_tokens) :
            self.word_to_id[token] = index
            self.id_to_word[index] = token
            self.vocab_size += 1

        set_word = []
        for text in texts:
            for word in text.lower().split():
                set_word.append(word)

        set_word = sorted(set_word)
        for word in set_word:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        res = []
        text_list = text.lower().split()
        for word in text_list :
            if word not in self.word_to_id :
                res.append(1)
            else :
                res.append(self.word_to_id[word])
        return res
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        res = []
        for id in ids :
            if id >= self.vocab_size :
                id = 1
            res.append(self.id_to_word[id])
        ans = " ".join(res)
        return ans