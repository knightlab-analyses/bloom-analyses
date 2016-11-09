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


def _naive_mean_permutation_test(mat, cats, permutations=1000):
    """
    mat: numpy 2-d matrix
         columns: features (e.g. OTUs)
         rows: samples
         matrix of features
    cats: numpy array
         Array of categories to run group signficance on

    Note: only works on binary classes now

    Returns
    =======
    test_stats:
        List of mean test statistics
    pvalues:
        List of corrected p-values

    This module will conduct a mean permutation test using
    the naive approach
    """
    def _mean_test(values, cats):
        # calculates mean for binary categories
        return abs(values[cats == 0].mean()-values[cats == 1].mean())

    rows, cols = mat.shape
    pvalues = np.zeros(rows)
    test_stats = np.zeros(rows)
    for r in range(rows):
        values = mat[r, :].transpose()
        test_stat = _mean_test(values, cats)
        perm_stats = np.empty(permutations, dtype=np.float64)
        for i in range(permutations):
            perm_cats = np.random.permutation(cats)
            perm_stats[i] = _mean_test(values, perm_cats)
        p_value = ((perm_stats >= test_stat).sum() + 1.) / (permutations + 1.)
        pvalues[r] = p_value
        test_stats[r] = test_stat
    return test_stats, pvalues


def permutation_mean(table, grouping, permutations=1000, random_state=None):
    """ Conducts a fishers test on a contingency table.

    This module will conduct a mean permutation test using
    numpy matrix algebra.

    Parameters
    ----------
    table: pd.DataFrame
        Contingency table of where columns correspond to features
        and rows correspond to samples.
    grouping : pd.Series
        Vector indicating the assignment of samples to groups.  For example,
        these could be strings or integers denoting which group a sample
        belongs to.  It must be the same length as the samples in `table`.
        The index must be the same on `table` and `grouping` but need not be
        in the same order.
    permutations: int
         Number of permutations to calculate
    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.

    Return
    ------
    pd.DataFrame
        A table of features, their t-statistics and p-values
        `"m"` is the t-statistic.
        `"pvalue"` is the p-value calculated from the permutation test.

    Examples
    --------
    >>> from canvas.stats.permutation import fisher_mean
    >>> import pandas as pd
    >>> table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
    ...                       [9,  11, 12, 10, 10, 10, 10],
    ...                       [1,  11, 10, 11, 10, 5,  9],
    ...                       [22, 21, 9,  10, 10, 10, 10],
    ...                       [20, 22, 10, 10, 13, 10, 10],
    ...                       [23, 21, 14, 10, 10, 10, 10]],
    ...                      index=['s1','s2','s3','s4','s5','s6'],
    ...                      columns=['b1','b2','b3','b4','b5','b6','b7'])
    >>> grouping = pd.Series([0, 0, 0, 1, 1, 1],
    ...                      index=['s1','s2','s3','s4','s5','s6'])
    >>> results = fisher_mean(table, grouping,
    ...                       permutations=100, random_state=0)
    >>> results
                m    pvalue
    b1  14.333333  0.108911
    b2  10.333333  0.108911
    b3   0.333333  1.000000
    b4   0.333333  1.000000
    b5   1.000000  1.000000
    b6   1.666667  1.000000
    b7   0.333333  1.000000

    Notes
    -----
    Only works on binary classes.
    """

    mat, cats = check_table_grouping(table, grouping)
    perms = _init_reciprocal_perms(cats.values, permutations,
                                   random_state=random_state)

    m, p = _np_two_sample_mean_statistic(mat.values.T, perms)
    res = pd.DataFrame({'m': m, 'pvalue': p}, index=mat.columns)

    return res


def _init_reciprocal_perms(cats, permutations=1000, random_state=None):
    """
    Creates a reciprocal permutation matrix.
    This is to ease the process of division.

    cats: numpy.array
       List of binary class assignments
    permutations: int
       Number of permutations for permutation test

    Note: This can only handle binary classes now
    """
    random_state = check_random_state(random_state)
    num_cats = len(np.unique(cats))  # number of distinct categories
    c = len(cats)
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)),
                              dtype=np.float64))
    _samp_ones = np.array(np.ones(c), dtype=np.float64).transpose()
    for m in range(permutations+1):

        # Perform division to make mean calculation easier
        perms[:, 2*m] = copy_cats / float(copy_cats.sum())
        perms[:, 2*m+1] = (_samp_ones - copy_cats)
        perms[:, 2*m+1] /= float((_samp_ones - copy_cats).sum())
        random_state.shuffle(copy_cats)
    return perms


def _np_two_sample_mean_statistic(mat, perms):
    """
    Calculates a permutative mean statistic just looking at binary classes

    mat: numpy.ndarray or scipy.sparse.*
         Contingency table. Eolumns correspond to features (e.g. OTUs)
         and rows correspond to samples.

    perms: numpy.ndarray
         Permutative matrix.
         Columns correspond to  permutations of samples
         rows corresponds to features

    Note: only works on binary classes now

    Returns
    =======
    m:
        List of mean test statistics
    p:
        List of corrected p-values

    This module will conduct a mean permutation test using
    numpy matrix algebra
    """
    # Create a permutation matrix
    num_cats = 2  # number of distinct categories
    n_otus, c = perms.shape
    permutations = (c-num_cats) // num_cats

    # Perform matrix multiplication on data matrix
    # and calculate averages
    avgs = mat.dot(perms)
    # Calculate the mean statistic
    idx = np.arange(0, (permutations+1)*num_cats, num_cats)
    mean_stat = abs(avgs[:, idx+1] - avgs[:, idx])

    # Calculate the p-values
    cmps = (mean_stat[:, 1:].T >= mean_stat[:, 0]).T
    pvalues = (cmps.sum(axis=1) + 1.)/(permutations + 1.)

    m = np.ravel(mean_stat[:, 0])
    p = np.array(np.ravel(pvalues))
    return m, p
