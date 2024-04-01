from typing import Optional, List, Iterable, Tuple, Union

import numpy as np
from numpy import ndarray


def argsort_by_magnitude_with_negs_last(
    a: Union[ndarray, List, Iterable]
) -> ndarray:
    """
    """
    kind = "mergesort"
    argsort_1 = np.argsort(-a, kind=kind)  # Descending
    argsort_2 = np.argsort(np.abs(a[argsort_1]), kind=kind)
    return argsort_1[argsort_2]


def partial_log_likelihood(
        t: Union[ndarray, List, Iterable],
        pred: Union[ndarray, List, Iterable],
        weights: Optional[Union[ndarray, List, Iterable]] = None,
        sorter: Optional[Union[ndarray, List, Iterable]] = None,
) -> float:
    """
    TODO
    """
    assert len(t) == len(pred)

    exp_p_sum = 0
    n_data = len(t)
    if sorter is None:
        sorter = np.arange(n_data)
    sort_backwards = sorter[::-1]

    log_sum_exp_offset = 0.0
    for ii in range(n_data):
        this_pred = pred[sorter[n_data - ii - 1]]
        if this_pred > 0:
            log_sum_exp_offset = this_pred
            break

    prev_t = np.inf
    out = 0.0
    weights_sum = 0.0

    for n, idx in enumerate(sort_backwards):
        t_i = t[idx]
        p_i = pred[idx]
        w_i = 1.0 if weights is None else weights[idx]

        if abs(t_i) < abs(prev_t):

            exp_p_sum += w_i * np.exp(p_i - log_sum_exp_offset)
            jj = 1
            while n + jj < n_data and abs(t[sort_backwards[n + jj]]) == abs(t_i):
                idx_new = sort_backwards[n + jj]
                w_new = 1.0 if weights is None else weights[idx_new]
                exp_p_sum += w_new * np.exp(pred[idx_new] - log_sum_exp_offset)
                jj += 1

        if t_i > 0:
            out += w_i * (p_i - log_sum_exp_offset - np.log(exp_p_sum))
            weights_sum += w_i

        prev_t = t_i
    return out


def grad_partial_log_likelihood(
        t: Union[ndarray, List, Iterable],
        pred: Union[ndarray, List, Iterable],
        weights: Optional[Union[ndarray, List, Iterable]] = None,
        sorter: Optional[Union[ndarray, List, Iterable]] = None,
) -> Tuple[ndarray, ndarray]:
    """
    TODO
    """
    n_data = len(t)
    if sorter is None:
        sorter = np.arange(n_data)
    sorter_reverse = sorter[::-1]

    log_sum_exp_offset = 0.0
    for ii in range(n_data):
        this_pred = pred[sorter[n_data - ii - 1]]
        if this_pred > 0:
            log_sum_exp_offset = this_pred
            break

    exp_p_cum_sum = np.zeros(n_data, dtype=np.float64)
    prev_abs_t_ii = np.inf
    for n, idx in enumerate(sorter_reverse):
        abs_t_ii = abs(t[idx])
        w_ii = 1.0 if weights is None else weights[idx]
        if abs_t_ii < prev_abs_t_ii:
            exp_p_sum_ii = w_ii * np.exp(pred[idx] - log_sum_exp_offset)
            jj = 1
            while (n + jj < n_data) and abs_t_ii == abs(t[sorter_reverse[n + jj]]):
                idx_new = sorter_reverse[n + jj]
                w_new = 1.0 if weights is None else weights[idx_new]
                exp_p_sum_ii += w_new * np.exp(pred[idx_new] - log_sum_exp_offset)
                jj += 1

            exp_p_cum_sum[n] = exp_p_sum_ii + exp_p_cum_sum[n - 1]
        elif abs_t_ii == prev_abs_t_ii:
            exp_p_cum_sum[n] = exp_p_cum_sum[n - 1]
        else:
            print("Count:", n)
            print("Index:", idx)
            print("Previous magnitude:", prev_abs_t_ii)
            print("Current magnitude:", abs_t_ii)
            raise ValueError("t is not sorted.")

        prev_abs_t_ii = abs_t_ii
    exp_p_cum_sum = exp_p_cum_sum[::-1]

    r_k = 0.0
    s_k = 0.0
    prev_abs_t_i = -1
    grad = np.zeros(n_data, dtype=np.float64)
    hess = np.zeros(n_data, dtype=np.float64)
    for n, idx in enumerate(sorter):
        p_i = pred[idx]
        w_i = 1.0 if weights is None else weights[idx]
        exp_p_i_offset = w_i * np.exp(p_i - log_sum_exp_offset)
        t_i = t[idx]
        abs_t_i = np.abs(t_i)
        exp_p_sum = exp_p_cum_sum[n]

        if t_i > 0 and abs_t_i > prev_abs_t_i:
            n_ties = 1
            while n + n_ties < n_data and t[sorter[n + n_ties]] == t_i:
                w_tied_i = 1.0 if weights is None else weights[sorter[n + n_ties]]
                n_ties += int(w_tied_i)
            r_k += n_ties / exp_p_sum
            s_k += n_ties / exp_p_sum ** 2

        grad[idx] = w_i * ((t_i > 0) - exp_p_i_offset * r_k)
        hess[idx] = w_i * exp_p_i_offset * (exp_p_i_offset * s_k - r_k)
        prev_abs_t_i = abs_t_i

    return grad, hess


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

        if weights is not None:
            weights = np.asarray(weights)
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