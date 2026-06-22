import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # 1. Initialize a 1D NumPy array of zeros matching the length of the vocabulary
    vector = np.zeros(len(vocab), dtype=int)
    
    # 2. Convert the vocab list into a lookup dictionary (word -> index)
    # This prevents the "list indices must be integers" error and keeps lookups fast!
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    # 3. Count occurrences
    for token in tokens:
        if token in vocab_dict:
            idx = vocab_dict[token]
            vector[idx] += 1
            
    return vector