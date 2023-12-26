import os
from datasets import Block3DDataset

# Get all folders in worlds folder
all_worlds = os.listdir("worlds")

# Create a dataset for each world
for world in all_worlds:
    data_path = "worlds/" + world
    dataset_path = "datasets/" + world + ".pt"
    dataset = Block3DDataset(dataset_path)
    dataset.generate_pairs(2)
    dataset.save()
