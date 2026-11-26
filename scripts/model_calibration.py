from __future__ import annotations

import numpy as np


class SigmoidCalibratedModel:
    """Apply chronological Platt calibration to a fitted estimator."""

    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator

    @property
    def named_steps(self):
        return self.estimator.named_steps

    @property
    def feature_names_in_(self):
        return self.estimator.feature_names_in_

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, features):
        raw_probability = self.estimator.predict_proba(features)[:, 1]
        raw_probability = np.clip(raw_probability, 1e-6, 1 - 1e-6)
        logits = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(logits)[:, 1]
        return np.column_stack((1 - calibrated, calibrated))
