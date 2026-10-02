"""Scaffold-based splits. The training pool consists of frequent scaffolds, and the test pool of the other scaffolds."""

from typing import NamedTuple

import numpy as np
import pandas as pd
from rdkit import Chem

from src.features import SCAFFOLD_COL, SMILES_COL, featurize, fingerprints_by_smiles, max_tanimoto

# A molecule is kept only if its scaffold contains at least this fraction of its heavy atoms.
MIN_SCAFFOLD_HEAVY_ATOM_FRACTION = 2 / 3
# Seed for sampling molecules within each scaffold. It is fixed so that the pools are the same for all trials.
POOL_CAP_SEED = 42


class ScaffoldPools(NamedTuple):
    train_pool: pd.DataFrame
    test_pool: pd.DataFrame


class TrainIdTest(NamedTuple):
    train: pd.DataFrame
    id_test: pd.DataFrame


def build_scaffold_pools(
    molecules: pd.DataFrame,
    min_compounds_per_scaffold: int = 6,
    cap_per_scaffold: int = 6,
    should_featurize: bool = False,
) -> ScaffoldPools:
    if should_featurize:
        molecules = featurize(molecules)
    molecules = molecules[molecules.apply(_has_dominant_scaffold, axis=1)]
    scaffold_counts = molecules[SCAFFOLD_COL].value_counts()
    train_pool = molecules[molecules[SCAFFOLD_COL].map(scaffold_counts) >= min_compounds_per_scaffold]
    train_pool = train_pool.groupby(SCAFFOLD_COL).sample(n=cap_per_scaffold, random_state=POOL_CAP_SEED)
    test_pool = molecules[~molecules[SCAFFOLD_COL].isin(train_pool[SCAFFOLD_COL].unique())]
    return ScaffoldPools(train_pool.reset_index(drop=True), test_pool)


def select_ood_test(pools: ScaffoldPools, similarity_range: tuple[float, float] = (0.0, 0.3)) -> pd.DataFrame:
    fingerprints = fingerprints_by_smiles(pools.train_pool[SMILES_COL])
    low, high = similarity_range
    is_dissimilar = [low <= max_tanimoto(smiles, fingerprints) <= high for smiles in pools.test_pool[SMILES_COL]]
    return pools.test_pool[is_dissimilar].reset_index(drop=True)


def sample_train_id_test(
    train_pool: pd.DataFrame, n_train_scaffolds: int, random_state: int, id_test_per_scaffold: int = 1
) -> TrainIdTest:
    scaffolds = train_pool[SCAFFOLD_COL].unique()
    selected = np.random.RandomState(random_state).choice(scaffolds, n_train_scaffolds, replace=False)
    sampled = train_pool[train_pool[SCAFFOLD_COL].isin(selected)]
    rng = np.random.default_rng(random_state)
    train_parts, id_test_parts = [], []
    for _, group in sampled.groupby(SCAFFOLD_COL, sort=False):
        shuffled = group.iloc[rng.permutation(len(group))]
        id_test_parts.append(shuffled.iloc[:id_test_per_scaffold])
        train_parts.append(shuffled.iloc[id_test_per_scaffold:])
    return TrainIdTest(pd.concat(train_parts).reset_index(drop=True), pd.concat(id_test_parts).reset_index(drop=True))


def _has_dominant_scaffold(molecule: pd.Series) -> bool:
    # Acyclic molecules have no scaffold, and the column is empty for them.
    if pd.isna(molecule[SCAFFOLD_COL]):
        return False
    mol, scaffold = (Chem.MolFromSmiles(molecule[SMILES_COL]), Chem.MolFromSmiles(molecule[SCAFFOLD_COL]))
    if mol is None or scaffold is None:
        return False
    return scaffold.GetNumHeavyAtoms() / mol.GetNumHeavyAtoms() >= MIN_SCAFFOLD_HEAVY_ATOM_FRACTION
