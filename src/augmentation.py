"""Virtual molecules made by applying the MMP rules to the training pool, and their random sampling for each trial."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem

from src.data import Split
from src.features import SMILES_COL
from src.transforms import MMPTransformer

RULE_COL = "transform"
ORIG_SMILES_COL = "orig_smiles"
POOL_COLUMNS = [SMILES_COL, ORIG_SMILES_COL]
TRANSFORM_SEED = 42


def generate_mmp_pool(train_pool: pd.DataFrame, rules_file: str, n_jobs: int = 1) -> pd.DataFrame:
    rules = pd.read_csv(rules_file)[RULE_COL].tolist()
    parents = train_pool[SMILES_COL].tolist()
    batches = np.array_split(np.arange(len(parents)), max(1, min(n_jobs, len(parents))))
    records_per_batch = Parallel(n_jobs=n_jobs)(
        delayed(_products_of)([parents[index] for index in batch], rules) for batch in batches
    )
    pool = pd.DataFrame([record for records in records_per_batch for record in records], columns=POOL_COLUMNS)
    # A product generated from more than one parent is kept once, with the first parent.
    return pool.drop_duplicates(subset=[SMILES_COL]).reset_index(drop=True)


def _products_of(parents: list[str], rules: list[str]) -> list[tuple[str, str]]:
    transformer = MMPTransformer(rules)
    records = []
    for parent in parents:
        rng = np.random.RandomState(TRANSFORM_SEED)
        products = transformer.products(Chem.MolFromSmiles(parent), rng)
        records.extend((Chem.MolToSmiles(product), parent) for product in products)
    return records


def sample_virtual_molecules(pool: pd.DataFrame, split: Split, n_expansions: int, random_state: int) -> pd.DataFrame:
    candidates = pool[pool[ORIG_SMILES_COL].isin(split.train[SMILES_COL])]
    known = set(split.train[SMILES_COL]) | set(split.id_test[SMILES_COL]) | set(split.ood_test[SMILES_COL])
    candidates = candidates[~candidates[SMILES_COL].isin(known)]
    max_size = len(split.train) * n_expansions
    if len(candidates) > max_size:
        candidates = candidates.sample(n=max_size, random_state=random_state)
    return candidates.reset_index(drop=True)
