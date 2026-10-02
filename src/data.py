import os
from typing import NamedTuple

import pandas as pd

TRAIN_FILE = "train.csv"
ID_TEST_FILE = "id_test.csv"
OOD_TEST_FILE = "ood_test.csv"


class Split(NamedTuple):
    train: pd.DataFrame
    id_test: pd.DataFrame
    ood_test: pd.DataFrame


def load_split(seed_dir: str) -> Split:
    file_names = (TRAIN_FILE, ID_TEST_FILE, OOD_TEST_FILE)
    return Split(*(pd.read_csv(os.path.join(seed_dir, file_name)) for file_name in file_names))


def save_csv(molecules: pd.DataFrame, directory: str, file_name: str) -> None:
    os.makedirs(directory, exist_ok=True)
    molecules.to_csv(os.path.join(directory, file_name), index=False)
