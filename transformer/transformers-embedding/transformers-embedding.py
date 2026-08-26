import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    embeddings = embedding(tokens)
    scaled_embeddings = embeddings * (d_model**0.5)
    return scaled_embeddings