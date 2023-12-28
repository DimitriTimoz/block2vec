import os
import json

import torch
from torch.utils.data import Dataset

import numpy as np
from nbt.world import WorldFolder

from world_discover import get_block, parse_block_data

class Block3DDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.processed = False
        self.load(path)
        
        
    def generate_pairs(self, path, window_size=2):
        """
        Generate pairs of (target, context) blocks from the Minecraft data.
        Target is the center block.
        Context is the surrounding blocks and the biome.
        """
        
        # We use a sliding window to generate the pairs
        # The window size is the number of blocks around the center block
        name = os.path.basename(path)
        self.window_size = window_size
        print(f"Loading world {name} ...")
        world = WorldFolder(path)
        print(f"World {name} loaded")

        f = open("blocks_ids.json", "r")
        blocks_ids = dict(json.load(f))
        f = open("biomes_ids.json", "r")
        biomes_ids = dict(json.load(f))

        block_context = np.zeros((window_size * 2 + 1, window_size * 2 + 1, window_size * 2 + 1))
        biome_context = 0

        # Iterate over all the chunks in the world
        for chunk in world.iter_nbt():
            for section in chunk["sections"]:
                if "biomes" not in section.keys():
                    continue
                biomes = section["biomes"]
                biomes_palette = biomes["palette"]
                
                
                block_states = section["block_states"]
                block_palette = block_states["palette"]
                
                if len(biomes_palette) == 1:
                    biome_context = np.full((16, 16, 16), biomes_ids[str(biomes_palette[0])])
                    print("biome_context", biome_context)
                elif len(biomes_palette) > 1:
                    # Find the biome of the center block
                    print("biomes_palette", biomes_palette)
                    print("biomes", biomes["data"])
                    
                else:
                    continue
                
                if len(block_palette) == 1:
                    id = blocks_ids.get(str(block_palette[0]["Name"]))
                    if id is None:
                        print(f"Unknown block: {block_palette[0]}")
                        continue
                    # Take only 5% of full blocks
                    if np.random.rand() > 0.05:
                        continue
                    block_context = np.full((16, 16, 16), id)
                    print("block_context", block_context)
                elif len(block_palette) > 1:
                    print("block_palette", block_palette)
                    print("block_states", block_states["data"])
                    bits_per_value = max(int(np.ceil(np.log2(len(block_palette)))), 1)
                    block_states_indices = parse_block_data(block_states["data"], bits_per_value)
                    block_context = [block_palette[index] if index < len(block_palette) else None for index in block_states_indices]
                    print("block_context", block_context)
                    return
                else:
                    continue
        # Target: the blocks around the center block and the biome
        # the data will be one-hot encoded
        # Context: the center block
        # the data will be one-hot encoded
        
                
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
