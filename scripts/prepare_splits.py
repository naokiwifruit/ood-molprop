"""Build each task's train / ID test / OOD test splits from curated_data.csv.

    python -m scripts.prepare_splits --task-dirs data/tasks/density data/tasks/logd data/tasks/logs data/tasks/mp \\
        --train-sizes 50 100 200 300 --seeds $(seq 0 29) --overwrite

Output: data/tasks/<task>/n{size}/seed{seed}/{train,id_test,ood_test}.csv
"""

import argparse
import os
import shutil

import pandas as pd

from src.data import ID_TEST_FILE, OOD_TEST_FILE, TRAIN_FILE, save_csv
from src.features import SCAFFOLD_COL
from src.splits import ScaffoldPools, build_scaffold_pools, sample_train_id_test, select_ood_test

CURATED_FILE = "curated_data.csv"
TRAIN_POOL_FILE = "train_pool.csv"
TEST_POOL_FILE = "test_pool.csv"
# Each sampled scaffold gives TRAIN_PER_SCAFFOLD molecules to train.csv and ID_TEST_PER_SCAFFOLD to id_test.csv.
TRAIN_PER_SCAFFOLD = 5
ID_TEST_PER_SCAFFOLD = 1
CAP_PER_SCAFFOLD = TRAIN_PER_SCAFFOLD + ID_TEST_PER_SCAFFOLD
MIN_COMPOUNDS_PER_SCAFFOLD = CAP_PER_SCAFFOLD


def prepare_task(
    task_dir: str,
    train_sizes: list[int],
    seeds: list[int],
    similarity_range: tuple[float, float],
    should_featurize: bool,
) -> None:
    pools = _load_or_build_pools(task_dir, should_featurize)
    ood_test = select_ood_test(pools, similarity_range)
    print(f"  ood_test: {len(ood_test)} molecules, shared by every size and seed")
    for train_size in train_sizes:
        if train_size % TRAIN_PER_SCAFFOLD:
            raise ValueError(f"train size {train_size} is not a multiple of {TRAIN_PER_SCAFFOLD}")
        for seed in seeds:
            train, id_test = sample_train_id_test(
                pools.train_pool, train_size // TRAIN_PER_SCAFFOLD, seed, ID_TEST_PER_SCAFFOLD
            )
            seed_dir = os.path.join(task_dir, f"n{train_size}", f"seed{seed}")
            save_csv(train, seed_dir, TRAIN_FILE)
            save_csv(id_test, seed_dir, ID_TEST_FILE)
            save_csv(ood_test, seed_dir, OOD_TEST_FILE)
            print(
                f"  n{train_size} seed{seed}: train {len(train)} ({train[SCAFFOLD_COL].nunique()} scaffolds), "
                f"id_test {len(id_test)}"
            )


def _load_or_build_pools(task_dir: str, should_featurize: bool) -> ScaffoldPools:
    train_pool_path, test_pool_path = (os.path.join(task_dir, TRAIN_POOL_FILE), os.path.join(task_dir, TEST_POOL_FILE))
    if os.path.exists(train_pool_path) and os.path.exists(test_pool_path):
        print(f"  pools: reusing {TRAIN_POOL_FILE} and {TEST_POOL_FILE} (pass --overwrite to rebuild them)")
        return ScaffoldPools(pd.read_csv(train_pool_path), pd.read_csv(test_pool_path))
    print(f"  pools: building {TRAIN_POOL_FILE} and {TEST_POOL_FILE} from {CURATED_FILE}")
    molecules = pd.read_csv(os.path.join(task_dir, CURATED_FILE))
    pools = build_scaffold_pools(molecules, MIN_COMPOUNDS_PER_SCAFFOLD, CAP_PER_SCAFFOLD, should_featurize)
    save_csv(pools.train_pool, task_dir, TRAIN_POOL_FILE)
    save_csv(pools.test_pool, task_dir, TEST_POOL_FILE)
    return pools


def _remove_outputs(task_dir: str, train_sizes: list[int]) -> None:
    for file_name in (TRAIN_POOL_FILE, TEST_POOL_FILE):
        path = os.path.join(task_dir, file_name)
        if os.path.exists(path):
            os.remove(path)
    for train_size in train_sizes:
        shutil.rmtree(os.path.join(task_dir, f"n{train_size}"), ignore_errors=True)


def main(args: argparse.Namespace) -> None:
    for task_dir in args.task_dirs:
        print(f"\n{os.path.basename(os.path.normpath(task_dir))}")
        if args.overwrite:
            _remove_outputs(task_dir, args.train_sizes)
        prepare_task(task_dir, args.train_sizes, args.seeds, tuple(args.sim_range), args.featurize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train / ID test / OOD test splits from curated_data.csv")
    parser.add_argument("--task-dirs", nargs="+", required=True, help="task directories, e.g. data/tasks/density")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--sim-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=[0.0, 0.3],
        help="keep OOD test molecules whose nearest-neighbor Tanimoto similarity to the training pool is in [LO, HI]",
    )
    parser.add_argument("--featurize", action="store_true", help="also write the descriptor and fingerprint columns")
    parser.add_argument("--overwrite", action="store_true", help="remove the pools and splits before rebuilding")
    main(parser.parse_args())
