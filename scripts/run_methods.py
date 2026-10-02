"""Build the baseline, Ensemble, and MMP-PL models of each trial and report their R² on the ID and OOD test sets.

Run scripts/prepare_splits.py and scripts/prepare_mmp_augmentation.py first.

    python -m scripts.run_methods --task-dir data/tasks/density --train-sizes 100 --seeds 0 1 2 --out-dir results

Output: <out-dir>/<task>/n{size}/seed{seed}.json
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import r2_score

from src.candidates import CandidateEvaluation, FeatureMatrices
from src.data import TRAIN_FILE, Split, load_split
from src.features import LABEL_COL, SMILES_COL, FeatureType, feature_columns, featurize
from src.methods import Baseline, Ensemble, MMPPseudoLabeling

VIRTUAL_FILE = "virtual_molecules.csv"
# The script that writes each input file, named in the error when the file is missing.
SCRIPT_BY_INPUT_FILE = {TRAIN_FILE: "prepare_splits", VIRTUAL_FILE: "prepare_mmp_augmentation"}
FeatureColumns = dict[FeatureType, list[str]]


class SharedOODTest:
    """The OOD test set of a task is shared by every size and seed (see prepare_splits.py), so it is featurized once."""

    def __init__(self):
        self.smiles: list[str] | None = None
        self.featurized: pd.DataFrame | None = None

    def featurize(self, ood_test: pd.DataFrame) -> pd.DataFrame:
        smiles = ood_test[SMILES_COL].tolist()
        if self.smiles is None:
            self.smiles, self.featurized = smiles, featurize(ood_test)
        elif smiles != self.smiles:
            raise ValueError("the OOD test set differs from the first trial's; rebuild the splits with --overwrite")
        return self.featurized


def run_trial(task: str, seed_dir: str, seed: int, n_jobs: int, shared_ood_test: SharedOODTest) -> dict:
    _check_inputs(seed_dir)
    train, id_test, ood_test = load_split(seed_dir)
    split = Split(featurize(train), featurize(id_test), shared_ood_test.featurize(ood_test))
    virtual = featurize(pd.read_csv(os.path.join(seed_dir, VIRTUAL_FILE)))
    columns = {feature_type: feature_columns(split.train, feature_type, task) for feature_type in FeatureType}
    train_features, y = _matrices(split.train, columns), split.train[LABEL_COL].values

    print("  Tuning 15 candidates ...", flush=True)
    candidates = CandidateEvaluation(train_features, y, seed, n_jobs).run()
    baseline = Baseline(candidates, seed).fit(train_features, y)
    ensemble = Ensemble(candidates, baseline, seed).fit(train_features, y)
    virtual_features = _matrices(virtual, columns)
    mmp_pl = MMPPseudoLabeling(baseline, teacher=ensemble, seed=seed).fit(train_features, y, virtual_features)
    methods = {"baseline": baseline, "ensemble": ensemble, "mmp_pl": mmp_pl}

    return {
        "task": task,
        "seed": seed,
        "n_train": len(split.train),
        "n_virtual": len(virtual),
        "baseline": baseline.candidate.name,
        "candidates": [
            {
                "name": candidate.name,
                "heldout_rmse": candidate.heldout_rmse,
                "heldout_r2": candidate.heldout_r2,
                "params": candidate.params,
            }
            for candidate in candidates
        ],
        "ensemble_weights": {
            member.algorithm: float(weight) for member, weight in zip(ensemble.members, ensemble.weights)
        },
        "id_r2": _r2_scores(methods, split.id_test, columns),
        "ood_r2": _r2_scores(methods, split.ood_test, columns),
    }


def _check_inputs(seed_dir: str) -> None:
    for file_name, script in SCRIPT_BY_INPUT_FILE.items():
        path = os.path.join(seed_dir, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run scripts/{script}.py first.")


def _matrices(molecules: pd.DataFrame, columns: FeatureColumns) -> FeatureMatrices:
    return {feature_type: molecules[names].values for feature_type, names in columns.items()}


def _r2_scores(methods: dict, test: pd.DataFrame, columns: FeatureColumns) -> dict[str, float]:
    features, y_true = _matrices(test, columns), test[LABEL_COL].values
    return {name: float(r2_score(y_true, method.predict(features))) for name, method in methods.items()}


def print_result(result: dict) -> None:
    methods = list(result["id_r2"])
    print(f"  baseline: {result['baseline']}")
    print(f"  {'':8s}" + "".join(f"{method:>10s}" for method in methods))
    for test_name in ("id_r2", "ood_r2"):
        print(f"  {test_name:8s}" + "".join(f"{result[test_name][method]:10.3f}" for method in methods))


def main(args: argparse.Namespace) -> None:
    task = os.path.basename(os.path.normpath(args.task_dir))
    shared_ood_test = SharedOODTest()
    for train_size in args.train_sizes:
        for seed in args.seeds:
            print(f"\n{task} n{train_size} seed{seed}")
            seed_dir = os.path.join(args.task_dir, f"n{train_size}", f"seed{seed}")
            result = run_trial(task, seed_dir, seed, args.njobs, shared_ood_test)
            print_result(result)
            out_dir = os.path.join(args.out_dir, task, f"n{train_size}")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"seed{seed}.json"), "w") as file:
                json.dump(result, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and evaluate the baseline, Ensemble, and MMP-PL models")
    parser.add_argument("--task-dir", required=True, help="task directory, e.g. data/tasks/density")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--njobs", type=int, default=1, help="parallel workers for the hyperparameter search")
    main(parser.parse_args())
