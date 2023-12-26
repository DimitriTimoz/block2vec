import torch
import torch.nn as nn

class Block2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Block2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out
