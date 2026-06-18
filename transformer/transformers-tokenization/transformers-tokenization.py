import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        # Special tokens setup
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Initialize special tokens with fixed IDs
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(self.special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            
        self.vocab_size = len(self.special_tokens)
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words in sorted order.
        """
        unique_words = set()
        
        # Extract unique words from all training texts
        for text in texts:
            # Lowercase and split by whitespace
            words = text.lower().split()
            for word in words:
                if word not in self.word_to_id:  # Exclude special tokens if they appear in text
                    unique_words.add(word)
        
        # Sort unique words alphabetically to maintain deterministic IDs
        sorted_words = sorted(list(unique_words))
        
        # Append sorted words to the mappings
        for word in sorted_words:
            current_id = self.vocab_size
            self.word_to_id[word] = current_id
            self.id_to_word[current_id] = word
            self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            # Fallback to <UNK> (ID: 1) if the word isn't in the vocabulary
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            token_ids.append(token_id)
            
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for token_id in ids:
            # Reconstruct string, fallback to <UNK> if ID is out of bounds
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)
            
        return " ".join(words)