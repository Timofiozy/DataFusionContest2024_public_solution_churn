from typing import Optional, List, Iterable, Tuple, Union

import numpy as np
from numpy import ndarray

from data_fusion_contest_2024_churn.ext.objectives import (
    argsort_by_magnitude_with_negs_last,
    partial_log_likelihood,
    grad_partial_log_likelihood
)

class CoxMixin:
    def __init__(self, auto_sort=True):
        self.auto_sort = auto_sort
        self._arg_sorter = None

    def argsort_by_magnitude(self, target: Union[ndarray, List, Iterable]) -> ndarray:
        target = np.asarray(target)
        if self.auto_sort:
            return argsort_by_magnitude_with_negs_last(target)
        else:
            return np.arange(target.shape[0], dtype=int)


class CoxPHObjective(CoxMixin):
    def is_max_optimal(self):
        return False

    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        approxes = np.asarray(approxes, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
        sorter = self.argsort_by_magnitude(targets)

        grad, hess = grad_partial_log_likelihood(
            t=targets, pred=approxes, weights=weights, sorter=sorter
        )
        return list(zip(grad, hess))


class CoxPHMetric(CoxMixin):
    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        approx = np.asarray(approx, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if weight is not None:
            weight = np.asarray(weight, dtype=np.float64)

        sorter = self.argsort_by_magnitude(target)

        error_sum = -partial_log_likelihood(
            t=target, pred=approx, weights=weight, sorter=sorter
        )
        if weight is None:
            weight_sum = len(approx)
        else:
            weight_sum = np.sum(weight)
        return error_sum, weight_sum

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)