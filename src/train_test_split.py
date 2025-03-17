#!/bin/python

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse




def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and split dataset into train, validation, and test sets.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed dataset.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test_split', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--val_split', type=float, default=0.1, help="Proportion of the dataset to include in the validation split.")
    return parser.parse_args()















def main():
        args = parse_arguments()
        DATASET_DIR = args.dataset_dir
        OUTPUT_DIR = args.output_dir
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

        crosswalks = np.array(glob.glob(os.path.join(DATASET_DIR, "crosswalk", "*.png")))
        not_crosswalks = np.array(glob.glob(os.path.join(DATASET_DIR, "not_crosswalk", "*.png")))

        data_size = len(crosswalks)
        test_size = int(data_size * args.test_split)
        val_size = int((data_size - test_size) * args.val_split)
        train_size = data_size - test_size - val_size

        indices = np.arange(data_size)
        np.random.shuffle(indices)

        train_idxs = indices[:train_size]
        val_idxs = indices[train_size:train_size + val_size]
        test_idxs = indices[train_size + val_size:]

        train_cswk, train_ncswk = crosswalks[train_idxs], not_crosswalks[train_idxs]
        val_cswk, val_ncswk = crosswalks[val_idxs], not_crosswalks[val_idxs]
        test_cswk, test_ncswk = crosswalks[test_idxs], not_crosswalks[test_idxs]

        train = {"crosswalk": train_cswk, "not_crosswalk": train_ncswk}
        val = {"crosswalk": val_cswk, "not_crosswalk": val_ncswk}
        test = {"crosswalk": test_cswk, "not_crosswalk": test_ncswk}

        dataset = {"train": train, "validation": val, "test": test}

        create_dataset(OUTPUT_DIR, dataset)



if __name__ == "__main__":
    main()
