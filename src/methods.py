"""The baseline and the two approaches.

Each is built from the evaluated candidates and predicts from feature matrices.
"""

import numpy as np

from src.candidates import Candidate, FeatureMatrices, fit_candidate, predict
from src.models import ALGORITHMS

# Lower bound of the RMSE in the inverse-RMSE weights, to avoid division by zero.
RMSE_FLOOR = 1e-12


class CandidateModel:
    def __init__(self, candidate: Candidate, seed: int):
        self.candidate = candidate
        self.seed = seed
        self.model = None

    def predict(self, features: FeatureMatrices) -> np.ndarray:
        return predict(self.model, features[self.candidate.feature_type])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> "CandidateModel":
        self.model = fit_candidate(self.candidate, X, y, self.seed)
        return self


class Baseline(CandidateModel):
    def __init__(self, candidates: list[Candidate], seed: int):
        super().__init__(min(candidates, key=lambda candidate: candidate.heldout_rmse), seed)

    def fit(self, features: FeatureMatrices, y: np.ndarray) -> "Baseline":
        return self._fit(features[self.candidate.feature_type], y)


class Ensemble:
    def __init__(self, candidates: list[Candidate], baseline: Baseline, seed: int):
        self.feature_type = baseline.candidate.feature_type
        by_algorithm = {
            candidate.algorithm: candidate for candidate in candidates if candidate.feature_type == self.feature_type
        }
        self.members = [by_algorithm[algorithm] for algorithm in ALGORITHMS]
        self.weights = self._inverse_rmse_weights(self.members)
        self.seed = seed
        self.models = []

    def fit(self, features: FeatureMatrices, y: np.ndarray) -> "Ensemble":
        self.models = [fit_candidate(member, features[self.feature_type], y, self.seed) for member in self.members]
        return self

    def predict(self, features: FeatureMatrices) -> np.ndarray:
        X = features[self.feature_type]
        return np.column_stack([predict(model, X) for model in self.models]) @ self.weights

    @staticmethod
    def _inverse_rmse_weights(members: list[Candidate]) -> np.ndarray:
        rmse = np.array([member.heldout_rmse for member in members])
        # Members with a negative R², which are worse than predicting the mean, are excluded.
        # If all members are excluded, only the best one is kept.
        is_kept = np.array([member.heldout_r2 >= 0 for member in members])
        if not is_kept.any():
            is_kept[np.argmin(rmse)] = True
        weights = np.where(is_kept, 1 / np.maximum(rmse, RMSE_FLOOR), 0)
        return weights / weights.sum()


class MMPPseudoLabeling(CandidateModel):
    def __init__(self, baseline: Baseline, teacher: Ensemble, seed: int):
        super().__init__(baseline.candidate, seed)
        self.teacher = teacher

    def fit(self, features: FeatureMatrices, y: np.ndarray, virtual_features: FeatureMatrices) -> "MMPPseudoLabeling":
        feature_type = self.candidate.feature_type
        X = np.vstack([features[feature_type], virtual_features[feature_type]])
        return self._fit(X, np.concatenate([y, self.teacher.predict(virtual_features)]))
