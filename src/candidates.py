"""Cross-validation for model construction.

The 15 candidates are tuned on the training molecules, and their held-out RMSE and R² are computed from the
out-of-fold predictions of the selected hyperparameters.
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.features import FeatureType
from src.models import ALGORITHMS, N_FOLDS, make_pipeline

N_SEARCH = 20
SCORING = "neg_root_mean_squared_error"
FeatureMatrices = dict[FeatureType, np.ndarray]


class Candidate(NamedTuple):
    algorithm: str
    feature_type: FeatureType
    params: dict
    heldout_rmse: float
    heldout_r2: float

    @property
    def name(self) -> str:
        return f"{self.algorithm}_{self.feature_type.value}"


class CandidateEvaluation:
    def __init__(self, features: FeatureMatrices, y: np.ndarray, seed: int, n_jobs: int = 1):
        self.features = features
        self.y = y
        self.seed = seed
        self.n_jobs = n_jobs
        self.folds = KFold(N_FOLDS, shuffle=True, random_state=seed)

    def run(self) -> list[Candidate]:
        return [self._evaluate(algorithm, feature_type) for algorithm in ALGORITHMS for feature_type in FeatureType]

    def _evaluate(self, algorithm: str, feature_type: FeatureType) -> Candidate:
        X = self.features[feature_type]
        params = self._tune(algorithm, X)
        predictions = self._out_of_fold_predictions(algorithm, X, params)
        rmse, r2 = (root_mean_squared_error(self.y, predictions), r2_score(self.y, predictions))
        return Candidate(algorithm, feature_type, params, float(rmse), float(r2))

    def _tune(self, algorithm: str, X: np.ndarray) -> dict:
        pipeline, grid = make_pipeline(algorithm, *X.shape, self.seed)
        search = RandomizedSearchCV(
            pipeline, grid, n_iter=N_SEARCH, cv=self.folds, scoring=SCORING, n_jobs=self.n_jobs, random_state=self.seed
        )
        with suppressed_warnings():
            search.fit(X, self.y)
        return dict(search.best_params_)

    def _out_of_fold_predictions(self, algorithm: str, X: np.ndarray, params: dict) -> np.ndarray:
        predictions = np.zeros(len(self.y))
        for train_index, validation_index in self.folds.split(X):
            model = fit_pipeline(algorithm, X[train_index], self.y[train_index], params, self.seed)
            predictions[validation_index] = predict(model, X[validation_index])
        return predictions


def fit_candidate(candidate: Candidate, X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    return fit_pipeline(candidate.algorithm, X, y, candidate.params, seed)


def fit_pipeline(algorithm: str, X: np.ndarray, y: np.ndarray, params: dict, seed: int) -> Pipeline:
    pipeline = make_pipeline(algorithm, *X.shape, seed).pipeline.set_params(**params)
    with suppressed_warnings():
        return pipeline.fit(X, y)


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(X).ravel()


@contextmanager
def suppressed_warnings() -> Iterator[None]:
    # Convergence and deprecation warnings from the estimators are not needed here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
