# ood-molprop

Data sets and code for the paper

**Improving Molecular Property Prediction in Out-of-Distribution Chemical Space Using Ensemble Models and Data Augmentation**

The repository contains

1. **Data sets** for out-of-distribution (OOD) property prediction from small training sets. Each training set consists of analog series (Bemis–Murcko scaffolds) with five molecules per series; the OOD test set consists of molecules dissimilar to all training molecules, and the in-distribution (ID) test set of molecules sharing scaffolds with the training set.
2. **Virtual molecules** generated from the training molecules by matched molecular pair (MMP) transformation rules.
3. **The baseline and the two approaches** of the paper: a single model selected by cross-validation (baseline), an inverse-RMSE weighted ensemble of the candidate models (Ensemble), and the baseline retrained on the training molecules and the virtual molecules pseudo-labeled by the ensemble (MMP pseudo-labeling, MMP-PL).

The pre-trained neural networks evaluated in the paper ([ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry), [GROVER](https://github.com/tencent-ailab/grover), and [Uni-Mol](https://github.com/deepmodeling/Uni-Mol)) are not included, because each of them requires its own environment.

## Setup

```bash
conda env create -f environment.yml
conda activate ood-molprop
```

All the commands below are run in the repository root.

## Quick start

```bash
python -m scripts.prepare_splits --task-dirs data/tasks/density --train-sizes 100 --seeds 0 1 2
python -m scripts.prepare_mmp_augmentation --task-dirs data/tasks/density --train-sizes 100 --seeds 0 1 2 --njobs 4
python -m scripts.run_methods --task-dir data/tasks/density --train-sizes 100 --seeds 0 1 2 --njobs 4
```

The first two commands take seconds, and the third about a minute per trial. It prints the selected baseline and the R² of the three models on the ID and OOD test sets of each trial:

```
density n100 seed0
  baseline: PLS_descriptor
            baseline  ensemble    mmp_pl
  id_r2        0.854     0.829     0.802
  ood_r2       0.214     0.210     0.491
```

Building the data sets and virtual molecules of the paper (four tasks, training set sizes 50, 100, 200, and 300, 30 seeds) takes about a minute and 160 MB:

```bash
python -m scripts.prepare_splits --task-dirs data/tasks/density data/tasks/logd data/tasks/logs data/tasks/mp --seeds $(seq 0 29)
python -m scripts.prepare_mmp_augmentation --task-dirs data/tasks/density data/tasks/logd data/tasks/logs data/tasks/mp --seeds $(seq 0 29) --njobs 4
```

## Data

`data/tasks/<task>/curated_data.csv` contains the standardized molecules of each property data set (density, logD, logS, and melting point) with the columns `smiles`, `scaffold_smiles` (Bemis–Murcko scaffold), and `label`. Salts were removed, charges were neutralized, and duplicate entries were merged in advance. The scripts start from these files. The task name is taken from the directory name (`density`, `logd`, `logs`, or `mp`); for `logd` and `logs`, the descriptors related to logP are excluded from the molecular representations.

`data/transformation_rules/transformation_rules_top1k.csv` contains the 1,000 most frequent MMP transformation rules extracted from ChEMBL 31, written as reaction SMARTS.

The data sets are redistributed under their original licenses, which differ among the sources (see [data/LICENSES.md](data/LICENSES.md)). Commercial use of the density data set is not permitted.

## Scripts

The scripts are run with `python -m scripts.<name>`, and `--help` lists their options. The procedures are described in the Methods section of the paper; the summaries below say what each script reads and writes.

### prepare_splits

Scaffolds with at least six molecules form the **training pool**, and the other scaffolds the **test pool**. The **OOD test set**, shared by every training set size and seed, is the test-pool molecules whose nearest-neighbor similarity to the training pool (maximum Tanimoto similarity, ECFP4, 2048 bits) is 0.3 or lower. For each size and seed, scaffolds sampled from the training pool each contribute five molecules to the **training set** and one to the **ID test set**.

```
data/tasks/<task>/
├── train_pool.csv
├── test_pool.csv
└── n{50,100,200,300}/seed{0..29}/
    ├── train.csv
    ├── id_test.csv
    └── ood_test.csv
```

| Option | Default | Description |
|---|---|---|
| `--train-sizes` | `50 100 200 300` | Training set sizes (multiples of 5) |
| `--seeds` | required | One seed for each trial |
| `--sim-range LO HI` | `0.0 0.3` | Range of the nearest-neighbor similarity of the OOD test molecules |
| `--featurize` | off | Also write the descriptor and fingerprint columns |
| `--overwrite` | off | Rebuild the pools and the splits |

### prepare_mmp_augmentation

Applies the transformation rules to every molecule in the training pool once, keeping at most one product per rule and molecule (`train_pool_mmp.csv`), and then samples the **virtual molecules** of each trial from the products of its training molecules, excluding those in the training, ID test, or OOD test sets. The number of virtual molecules is the training set size times the expansion ratio.

Output: `data/tasks/<task>/n{size}/seed{seed}/virtual_molecules.csv` with the columns `smiles` and `orig_smiles` (the training molecule from which the product was generated).

| Option | Default | Description |
|---|---|---|
| `--n-expansions` | `10` | Expansion ratio: the number of virtual molecules as a multiple of the training set size |
| `--rules` | `data/transformation_rules/transformation_rules_top1k.csv` | Transformation rules |
| `--njobs` | `1` | Parallel workers for applying the rules |
| `--overwrite` | off | Rebuild the products and redraw the virtual molecules of every trial |

### run_methods

Builds and evaluates the three models of each trial:

- **Candidates**: five algorithms (PLS, SVR, random forest, XGBoost, and MLP) × three representations (RDKit descriptors, ECFP4, and ECFP4 count), tuned by 3-fold cross-validation on the training molecules. Each candidate gets a held-out RMSE and R² from its out-of-fold predictions.
- **Baseline**: the candidate with the lowest held-out RMSE.
- **Ensemble**: the five algorithms sharing the representation of the baseline, weighted by the inverse of their held-out RMSE. Members with a negative held-out R² are excluded.
- **MMP-PL**: the baseline retrained on the training molecules and the virtual molecules, whose pseudo-labels are assigned by the Ensemble.

Each trial writes `results/<task>/n{size}/seed{seed}.json` with the ID and OOD R² of the three models, the selected hyperparameters and held-out scores of the candidates, and the ensemble weights. Most of the running time is the hyperparameter search.

| Option | Default | Description |
|---|---|---|
| `--train-sizes` | `50 100 200 300` | Training set sizes to run |
| `--seeds` | required | Seeds to run |
| `--out-dir` | `results` | Where the JSON files go |
| `--njobs` | `1` | Parallel workers for the hyperparameter search |

## License

The source code is licensed under the [MIT License](LICENSE). The data in `data/` are subject to separate licenses (see [data/LICENSES.md](data/LICENSES.md)).

## Citation

```
[Citation to be added upon publication]
```
