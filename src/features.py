"""Molecular representations: RDKit descriptors, binary Morgan fingerprints, and count Morgan fingerprints."""

from enum import Enum

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

SMILES_COL = "smiles"
SCAFFOLD_COL = "scaffold_smiles"
LABEL_COL = "label"

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

DESCRIPTOR_NAMES = {name for name, _ in Descriptors.descList}
# Ipc overflows for some molecules.
ALWAYS_EXCLUDED_DESCRIPTORS = {"Ipc"}
# Descriptors whose names contain these strings may leak the target property of the task.
LEAKAGE_PATTERNS_BY_TASK = {"logd": ["logp"], "logs": ["logp"]}


class FeatureType(str, Enum):
    DESCRIPTOR = "descriptor"
    FP = "fp"
    COUNTFP = "countfp"


FINGERPRINT_COLUMN_PREFIX = {FeatureType.FP: "fp_", FeatureType.COUNTFP: "countfp_"}


def calculate_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    return {name: function(mol) for name, function in Descriptors.descList}


def binary_fingerprint(smiles: str) -> np.ndarray:
    return _MORGAN_GENERATOR.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))


def count_fingerprint(smiles: str) -> np.ndarray:
    return _MORGAN_GENERATOR.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))


def fingerprints_by_smiles(smiles_list: list[str]) -> dict[str, DataStructs.ExplicitBitVect]:
    return {smiles: _MORGAN_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list}


def max_tanimoto(smiles: str, fingerprints: dict[str, DataStructs.ExplicitBitVect]) -> float:
    fingerprint = _MORGAN_GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))
    return max(DataStructs.TanimotoSimilarity(fingerprint, other) for other in fingerprints.values())


def featurize(molecules: pd.DataFrame) -> pd.DataFrame:
    smiles_list = molecules[SMILES_COL].tolist()
    records = [calculate_descriptors(smiles) for smiles in tqdm(smiles_list, desc="Featurizing")]
    descriptors = pd.DataFrame(records, index=molecules.index)
    binary = _fingerprint_frame([binary_fingerprint(smiles) for smiles in smiles_list], FeatureType.FP, molecules.index)
    counts = _fingerprint_frame(
        [count_fingerprint(smiles) for smiles in smiles_list], FeatureType.COUNTFP, molecules.index
    )
    # Molecules for which a descriptor could not be computed are dropped.
    return pd.concat([molecules, descriptors, binary, counts], axis=1).dropna(axis=0)


def _fingerprint_frame(fingerprints: list[np.ndarray], feature_type: FeatureType, index: pd.Index) -> pd.DataFrame:
    columns = [f"{FINGERPRINT_COLUMN_PREFIX[feature_type]}{bit}" for bit in range(MORGAN_BITS)]
    return pd.DataFrame(np.array(fingerprints), columns=columns, index=index)


def feature_columns(molecules: pd.DataFrame, feature_type: FeatureType, task: str | None = None) -> list[str]:
    if feature_type == FeatureType.DESCRIPTOR:
        excluded = ALWAYS_EXCLUDED_DESCRIPTORS | _leaking_descriptors(molecules.columns, task)
        return [column for column in molecules.columns if column in DESCRIPTOR_NAMES and column not in excluded]
    prefix = FINGERPRINT_COLUMN_PREFIX[feature_type]
    return [column for column in molecules.columns if column.startswith(prefix)]


def _leaking_descriptors(columns: pd.Index, task: str | None) -> set[str]:
    patterns = LEAKAGE_PATTERNS_BY_TASK.get(task, [])
    return {column for column in columns if any(pattern in column.lower() for pattern in patterns)}
