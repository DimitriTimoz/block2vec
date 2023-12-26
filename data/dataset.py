import torch
from torch.utils.data import Dataset

import os

class Block3DDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.processed = False
        self.load(path)
        
        
    def generate_pairs(self, window_size=2):
        """
        Generate pairs of (target, context) blocks from the Minecraft data.
        Target is the center block.
        Context is the surrounding blocks and the biome.
        """
        
        # We use a sliding window to generate the pairs
        # The window size is the number of blocks around the center block
        self.window_size = window_size
        
        self.processed = True

    def __len__(self):
        """
        Return the number of pairs in the dataset.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return a pair (target, context) at the given index.
        """
        target, context = self.pairs[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)
    
    def save(self):
        """
        Save the dataset
        """
        torch.save(self, self.path)
        
    def load(self, path):
        """
        Load the dataset
        """
        if os.path.exists(path):
            self = torch.load(path)

# Exemple d'utilisation
# Remplacer 'your_data' par vos données réelles et 'window_size' par la taille de fenêtre désirée
dataset = Block3DDataset("your_data, window_size=5")
