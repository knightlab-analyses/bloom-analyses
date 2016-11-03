from __future__ import division
import numpy as np
import pandas as pd
import copy
from scipy.stats import ttest_ind, f_oneway
from scipy._lib._util import check_random_state


def check_table_grouping(table, grouping):
    if not isinstance(table, pd.DataFrame):
        raise TypeError('`table` must be a `pd.DataFrame`, '
                        'not %r.' % type(table).__name__)
    if not isinstance(grouping, pd.Series):
        raise TypeError('`grouping` must be a `pd.Series`,'
                        ' not %r.' % type(grouping).__name__)

    if (grouping.isnull()).any():
        raise ValueError('Cannot handle missing values in `grouping`.')

    if (table.isnull()).any().any():
        raise ValueError('Cannot handle missing values in `table`.')

    groups, _grouping = np.unique(grouping, return_inverse=True)
    grouping = pd.Series(_grouping, index=grouping.index)
    num_groups = len(groups)
    if num_groups == len(grouping):
        raise ValueError(
            "All values in `grouping` are unique. This method cannot "
            "operate on a grouping vector with only unique values (e.g., "
            "there are no 'within' variance because each group of samples "
            "contains only a single sample).")
    if num_groups == 1:
        raise ValueError(
            "All values the `grouping` are the same. This method cannot "
            "operate on a grouping vector with only a single group of samples"
            "(e.g., there are no 'between' variance because there is only a "
            "single group).")
    table_index_len = len(table.index)
    grouping_index_len = len(grouping.index)
    mat, cats = table.align(grouping, axis=0, join='inner')
    if (len(mat) != table_index_len or len(cats) != grouping_index_len):
        raise ValueError('`table` index and `grouping` '
                         'index must be consistent.')
    return mat, cats



def _init_categorical_perms(cats, permutations=1000, random_state=None):
    """
    Creates a reciprocal permutation matrix

    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    random_state = check_random_state(random_state)
    c = len(cats)
    num_cats = len(np.unique(cats))  # Number of distinct categories
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)),
                              dtype=np.float64))
    for m in range(permutations+1):
        for i in range(num_cats):
            perms[:, num_cats*m+i] = (copy_cats == i).astype(np.float64)
        random_state.shuffle(copy_cats)
    return perms


def _naive_f_permutation_test(mat, cats, permutations=1000):
    """
    Performs a 1-way ANOVA.

    F = sum( MS_i for all i) /  MSE

    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a F permutation test using
    the naive approach
    """

    def _f_test(values, cats):
        # calculates t statistic for binary categories
        groups = []
        groups = [values[cats == k] for k in set(cats)]
        F, _ = f_oneway(*groups)
        return abs(F)

    rows, cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r, :].transpose()
        test_stat = _f_test(values, cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _f_test(values, perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    return test_stats, pvalues


def _np_k_sample_f_statistic(mat, cats, perms):
    """
    Calculates a permutative one way F test

    mat: numpy.array
         The contingency table.
         Columns correspond to features (e.g. OTUs)
         and rows correspond to  samples
    cat : numpy.array
         Vector of categories.
    perms: numpy.array
         Permutative matrix. Columns correspond to permutations
         of samples rows corespond to features

    Returns
    =======
    test_stats:
        List of f-test statistics
    pvalues:
        List of p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    # Create a permutation matrix
    num_cats = len(np.unique(cats))  # Number of distinct categories
    n_samp, c = perms.shape
    permutations = (c-num_cats) / num_cats

    mat2 = np.multiply(mat, mat)

    S = mat.sum(axis=1)
    SS = mat2.sum(axis=1)
    sstot = SS - np.multiply(S, S) / float(n_samp)
    # Create index to sum the ssE together
    _sum_idx = _init_categorical_perms(
        np.arange((permutations + 1) * num_cats, dtype=np.int32) // num_cats,
        permutations=0)

    # Perform matrix multiplication on data matrix
    # and calculate sums and squared sums and sum of squares
    _sums = np.dot(mat, perms)
    _sums2 = np.dot(np.multiply(mat, mat), perms)
    tot = perms.sum(axis=0)
    ss = _sums2 - np.multiply(_sums, _sums)/tot
    sserr = np.dot(ss, _sum_idx)
    sstrt = (sstot - sserr.T).T

    dftrt = num_cats-1
    dferr = np.dot(tot, _sum_idx) - num_cats
    f_stat = (sstrt / dftrt) / (sserr / dferr)

    cmps = f_stat[:, 1:].T >= f_stat[:, 0]
    pvalues = (cmps.sum(axis=0) + 1.) / (permutations + 1.)
    f = np.array(np.ravel(f_stat[:, 0]))
    p = np.array(np.ravel(pvalues))
    return f, p
