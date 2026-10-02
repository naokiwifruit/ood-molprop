"""Generate the virtual molecules of each trial.

The MMP rules are applied to the training pool once, and the virtual molecules are sampled for each trial.

Run scripts/prepare_splits.py first.

    python -m scripts.prepare_mmp_augmentation --task-dirs data/tasks/density data/tasks/logd data/tasks/logs \\
        data/tasks/mp --train-sizes 50 100 200 300 --seeds $(seq 0 29)

Output: data/tasks/<task>/train_pool_mmp.csv and data/tasks/<task>/n{size}/seed{seed}/virtual_molecules.csv
"""

import argparse
import os

import pandas as pd

from src.augmentation import generate_mmp_pool, sample_virtual_molecules
from src.data import load_split, save_csv
from src.features import SMILES_COL

DEFAULT_RULES = "data/transformation_rules/transformation_rules_top1k.csv"
TRAIN_POOL_FILE = "train_pool.csv"
POOL_FILE = "train_pool_mmp.csv"
VIRTUAL_FILE = "virtual_molecules.csv"


def prepare_task(
    task_dir: str,
    train_sizes: list[int],
    seeds: list[int],
    n_expansions: int,
    rules_file: str,
    n_jobs: int,
    should_overwrite: bool,
) -> None:
    pool = _load_or_build_pool(task_dir, rules_file, n_jobs, should_overwrite)
    n_skipped = 0
    for train_size in train_sizes:
        for seed in seeds:
            seed_dir = os.path.join(task_dir, f"n{train_size}", f"seed{seed}")
            if os.path.exists(os.path.join(seed_dir, VIRTUAL_FILE)) and not should_overwrite:
                n_skipped += 1
                continue
            virtual = sample_virtual_molecules(pool, load_split(seed_dir), n_expansions, random_state=seed)
            save_csv(virtual, seed_dir, VIRTUAL_FILE)
            print(f"  n{train_size} seed{seed}: {len(virtual)} virtual molecules")
    if n_skipped:
        print(f"  skipped {n_skipped} trials whose {VIRTUAL_FILE} exists (pass --overwrite to redraw them)")


def _load_or_build_pool(task_dir: str, rules_file: str, n_jobs: int, should_overwrite: bool) -> pd.DataFrame:
    pool_path = os.path.join(task_dir, POOL_FILE)
    if os.path.exists(pool_path) and not should_overwrite:
        pool = pd.read_csv(pool_path)
        print(f"  MMP pool: reusing {POOL_FILE} ({len(pool)} molecules; pass --overwrite to rebuild it)")
        return pool
    train_pool_path = os.path.join(task_dir, TRAIN_POOL_FILE)
    if not os.path.exists(train_pool_path):
        raise FileNotFoundError(f"{train_pool_path} not found. Run scripts/prepare_splits.py first.")
    train_pool = pd.read_csv(train_pool_path, usecols=[SMILES_COL])
    pool = generate_mmp_pool(train_pool, rules_file, n_jobs)
    save_csv(pool, task_dir, POOL_FILE)
    print(f"  MMP pool: {len(pool)} molecules from {len(train_pool)} parents")
    return pool


def main(args: argparse.Namespace) -> None:
    for task_dir in args.task_dirs:
        print(f"\n{os.path.basename(os.path.normpath(task_dir))}")
        prepare_task(task_dir, args.train_sizes, args.seeds, args.n_expansions, args.rules, args.njobs, args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the virtual molecules of each trial by MMP transformation")
    parser.add_argument("--task-dirs", nargs="+", required=True, help="task directories, e.g. data/tasks/density")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--n-expansions",
        type=int,
        default=10,
        help="virtual molecules per trial as a multiple of the training set size",
    )
    parser.add_argument("--rules", default=DEFAULT_RULES, help="transformation rules CSV")
    parser.add_argument("--njobs", type=int, default=1, help="parallel workers for the MMP pool")
    parser.add_argument("--overwrite", action="store_true", help="rebuild the MMP pool and every trial's draw")
    main(parser.parse_args())
