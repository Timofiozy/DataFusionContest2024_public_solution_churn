# distutils: language=c
cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()


def argsort_by_magnitude_with_negs_last(
    cnp.ndarray a
):
    """
    TODO
    """
    cdef cnp.ndarray argsort_1 = np.argsort(-a, kind="mergesort")
    cdef cnp.ndarray argsort_2 = np.argsort(np.abs(a[argsort_1]), kind="mergesort")
    return argsort_1[argsort_2]


@cython.boundscheck(False)
@cython.wraparound(False)
def partial_log_likelihood(
    cnp.ndarray t,
    cnp.ndarray pred,
    cnp.ndarray weights,
    cnp.ndarray sorter,
):
    """
    TODO
    """
    assert len(t) == len(pred)

    cdef long n_data = len(t)
    if sorter is None:
        sorter = np.arange(n_data)
    cdef cnp.ndarray sorter_reverse = sorter[::-1]

    cdef cnp.float64_t[:] t_view = t
    cdef cnp.float64_t[:] pred_view = pred
    cdef cnp.float64_t[:] weights_view = weights
    cdef long[:] sorter_view = sorter
    cdef long[:] sorter_reverse_view = sorter_reverse

    cdef cnp.float64_t log_sum_exp_offset = 0.0
    cdef long ii
    for ii in range(n_data):
        this_pred = pred_view[sorter_view[n_data - ii - 1]]
        if this_pred > 0:
            log_sum_exp_offset = this_pred
            break

    cdef cnp.float64_t prev_t = np.inf
    cdef cnp.float64_t out = 0.0
    cdef cnp.float64_t weights_sum = 0.0

    cdef cnp.float64_t exp_p_sum = 0.0
    cdef cnp.float64_t t_i
    cdef cnp.float64_t p_i
    cdef cnp.float64_t w_i
    cdef long jj
    cdef long idx_new
    cdef cnp.float64_t w_new
    cdef long n
    cdef long idx
    for n in range(len(sorter_reverse)):
        idx = sorter_reverse_view[n]
        t_i = t_view[idx]
        p_i = pred_view[idx]
        w_i = 1.0 if weights is None else weights_view[idx]

        if abs(t_i) < abs(prev_t):
            exp_p_sum += w_i * np.exp(p_i - log_sum_exp_offset)
            jj = 1
            while n + jj < n_data and abs(t_view[sorter_reverse_view[n + jj]]) == abs(t_i):
                idx_new = sorter_reverse_view[n + jj]
                w_new = 1.0 if weights is None else weights_view[idx_new]
                exp_p_sum += w_new * np.exp(pred_view[idx_new] - log_sum_exp_offset)
                jj += 1

        if t_i > 0:
            # Computing log likelihood, not its negative
            out += w_i * (p_i - log_sum_exp_offset - np.log(exp_p_sum))
            weights_sum += w_i

        prev_t = t_i

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def grad_partial_log_likelihood(
    cnp.ndarray t,
    cnp.ndarray pred,
    cnp.ndarray weights,
    cnp.ndarray sorter,
):
    """
    TODO
    """
    cdef long n_data = len(t)
    if sorter is None:
        sorter = np.arange(n_data)
    cdef cnp.ndarray sorter_reverse = sorter[::-1]

    cdef cnp.float64_t[:] t_view = t
    cdef cnp.float64_t[:] pred_view = pred
    cdef cnp.float64_t[:] weights_view = weights
    cdef long[:] sorter_view = sorter
    cdef long[:] sorter_reverse_view = sorter_reverse

    cdef cnp.float64_t log_sum_exp_offset = 0.0
    cdef long ii
    cdef cnp.float64_t this_pred
    for ii in range(n_data):
        this_pred = pred_view[sorter_view[n_data - ii - 1]]
        if this_pred > 0:
            log_sum_exp_offset = this_pred
            break

    cdef cnp.ndarray exp_p_cum_sum = np.zeros(n_data, dtype=np.float64)
    cdef cnp.float64_t[:] exp_p_cum_sum_view = exp_p_cum_sum
    cdef cnp.float64_t prev_abs_t_ii = np.inf
    cdef cnp.float64_t abs_t_ii
    cdef cnp.float64_t w_ii
    cdef cnp.float64_t exp_p_sum_ii
    cdef long jj
    cdef long idx_new
    cdef cnp.float64_t w_new
    cdef long idx
    cdef long n
    for n in range(len(sorter_reverse)):
        idx = sorter_reverse_view[n]
        abs_t_ii = abs(t_view[idx])
        w_ii = 1.0 if weights is None else weights_view[idx]
        if abs_t_ii < prev_abs_t_ii:
            exp_p_sum_ii = w_ii * np.exp(pred_view[idx] - log_sum_exp_offset)
            # Include all values that are tied as well (if they exist)
            jj = 1
            while (n + jj < n_data) and abs_t_ii == abs(t_view[sorter_reverse_view[n + jj]]):
                idx_new = sorter_reverse_view[n + jj]
                w_new = 1.0 if weights is None else weights_view[idx_new]
                exp_p_sum_ii += w_new * np.exp(pred_view[idx_new] - log_sum_exp_offset)
                jj += 1

            exp_p_cum_sum_view[n] = exp_p_sum_ii + exp_p_cum_sum_view[n - 1]
        elif abs_t_ii == prev_abs_t_ii:
            exp_p_cum_sum_view[n] = exp_p_cum_sum_view[n - 1]
        else:
            print("Count:", n)
            print("Index:", idx)
            print("Previous magnitude:", prev_abs_t_ii)
            print("Current magnitude:", abs_t_ii)
            raise ValueError("t is not sorted.")

        prev_abs_t_ii = abs_t_ii
    exp_p_cum_sum = exp_p_cum_sum[::-1]

    cdef cnp.float64_t r_k = 0.0
    cdef cnp.float64_t s_k = 0.0
    cdef cnp.float64_t prev_abs_t_i = -1
    cdef cnp.ndarray grad = np.zeros(n_data, dtype=np.float64)
    cdef cnp.ndarray hess = np.zeros(n_data, dtype=np.float64)
    cdef cnp.float64_t[:] grad_view = grad
    cdef cnp.float64_t[:] hess_view = hess
    cdef cnp.float64_t p_i
    cdef cnp.float64_t w_i
    cdef cnp.float64_t exp_p_i_offset
    cdef cnp.float64_t t_i
    cdef cnp.float64_t abs_t_i
    cdef cnp.float64_t exp_p_sum
    cdef long n_ties
    cdef cnp.float64_t w_tied_i
    for n in range(len(sorter)):
        idx = sorter_view[n]
        p_i = pred_view[idx]
        w_i = 1.0 if weights is None else weights_view[idx]
        exp_p_i_offset = w_i * np.exp(p_i - log_sum_exp_offset)
        t_i = t_view[idx]
        abs_t_i = abs(t_i)
        exp_p_sum = exp_p_cum_sum_view[n]

        if t_i > 0 and abs_t_i > prev_abs_t_i:
            n_ties = 1
            while n + n_ties < n_data and t_view[sorter_view[n + n_ties]] == t_i:
                w_tied_i = 1.0 if weights is None else weights_view[sorter_view[n + n_ties]]
                n_ties += int(w_tied_i)
            r_k += n_ties / exp_p_sum
            s_k += n_ties / exp_p_sum ** 2

        grad_view[idx] = w_i * ((t_i > 0) - exp_p_i_offset * r_k)
        hess_view[idx] = w_i * exp_p_i_offset * (exp_p_i_offset * s_k - r_k)
        prev_abs_t_i = abs_t_i

    return grad, hess

