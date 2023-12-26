import os
import concurrent.futures
from dataset import Block3DDataset

def process_world(world):
    worlds_folder = "worlds"
    datasets_folder = "datasets"

    data_path = os.path.join(worlds_folder, world)
    dataset_path = os.path.join(datasets_folder, world + ".pt")
    dataset = Block3DDataset(dataset_path)
    dataset.generate_pairs(data_path, window_size=2)
    dataset.save()

def main():
    worlds_folder = "worlds"
    all_worlds = os.listdir(worlds_folder)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_world, world) for world in all_worlds}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
