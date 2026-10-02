"""The five learning algorithms and their hyperparameter grids.

The pipeline standardizes the features before fitting.
"""

from collections.abc import Callable
from typing import NamedTuple

import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

N_FOLDS = 3
MAX_PLS_COMPONENTS = 10
SCALER_STEP = "scaler"
MODEL_STEP = "model"


class AlgorithmSpec(NamedTuple):
    estimator: BaseEstimator
    grid: dict[str, list]


class SearchSpace(NamedTuple):
    pipeline: Pipeline
    grid: dict[str, list]


def pls(n_molecules: int, n_features: int, random_state: int) -> AlgorithmSpec:
    # The number of PLS components cannot exceed the number of molecules in a CV fold.
    max_components = min(MAX_PLS_COMPONENTS, n_features, n_molecules // N_FOLDS)
    return AlgorithmSpec(PLSRegression(), {"n_components": list(range(1, max_components))})


def svr(n_molecules: int, n_features: int, random_state: int) -> AlgorithmSpec:
    grid = {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.1, 1, 10, 100]}
    return AlgorithmSpec(SVR(), grid)


def random_forest(n_molecules: int, n_features: int, random_state: int) -> AlgorithmSpec:
    grid = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [10, 25, 50],
        "min_samples_split": [2, 16, 32],
        "min_samples_leaf": [1, 16, 32],
        "max_features": ["sqrt", "log2"],
    }
    return AlgorithmSpec(RandomForestRegressor(random_state=random_state), grid)


def xgboost(n_molecules: int, n_features: int, random_state: int) -> AlgorithmSpec:
    grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6, 9],
        "min_child_weight": [1, 3, 5],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.6, 0.8, 1.0],
        "gamma": [0, 0.2, 0.4],
        "reg_alpha": [0, 1, 10],
        "reg_lambda": [0, 1, 10],
    }
    return AlgorithmSpec(xgb.XGBRegressor(random_state=random_state), grid)


def mlp(n_molecules: int, n_features: int, random_state: int) -> AlgorithmSpec:
    grid = {
        "hidden_layer_sizes": [(50,), (50, 50), (50, 50, 50)],
        "alpha": [0, 0.01, 0.1],
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "max_iter": [50, 200],
    }
    estimator = MLPRegressor(activation="relu", solver="adam", batch_size="auto", random_state=random_state)
    return AlgorithmSpec(estimator, grid)


ALGORITHMS: dict[str, Callable[[int, int, int], AlgorithmSpec]] = {
    "PLS": pls,
    "SVR": svr,
    "RF": random_forest,
    "XGBoost": xgboost,
    "MLP": mlp,
}


def make_pipeline(algorithm: str, n_molecules: int, n_features: int, random_state: int) -> SearchSpace:
    spec = ALGORITHMS[algorithm](n_molecules, n_features, random_state)
    pipeline = Pipeline([(SCALER_STEP, StandardScaler()), (MODEL_STEP, spec.estimator)])
    return SearchSpace(pipeline, {f"{MODEL_STEP}__{name}": values for name, values in spec.grid.items()})
