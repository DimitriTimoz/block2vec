import torch
import torch.nn as nn
import torch.optim as optim

class Block2Vec(nn.Module):
    def __init__(self, collection_size, embedding_dim):
        super(Block2Vec, self).__init__()
        self.embeddings = nn.Embedding(collection_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, collection_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs
