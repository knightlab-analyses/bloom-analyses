import numpy as np
import pandas as pd

import unittest
import numpy.testing as np_test

from bloom.stats.permutation import (_init_categorical_perms,
                                     _np_k_sample_f_statistic,
                                     _naive_f_permutation_test,
                                     check_table_grouping)
from skbio.util._testing import assert_data_frame_almost_equal


class CheckTableGroupingTests(unittest.TestCase):
    def setUp(self):
        # Basic count data with 2 groupings
        self.table1 = pd.DataFrame([
            [10, 10, 10, 20, 20, 20],
            [11, 12, 11, 21, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 13, 10, 10, 10, 12]]).T
        
        self.badcats1 = pd.Series([0, 0, 0, 1, np.nan, 1])
        self.badcats2 = pd.Series([0, 0, 0, 0, 0, 0])
        self.badcats3 = pd.Series([0, 0, 1, 1])
        self.badcats4 = pd.Series(range(len(self.table1)))
        self.badcats5 = pd.Series([1]*len(self.table1))

    def test_fail_missing(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats1)

    def test_fail_groups(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats2)

    def test_fail_size_mismatch(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats3)

    def test_fail_group_unique(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats4)

    def test_fail_1_group(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats5)
class TestPermutation(unittest.TestCase):

    def test_init_perms(self):
        cats = np.array([0, 1, 2, 0, 0, 2, 1])
        perms = _init_categorical_perms(cats, permutations=0)
        np_test.assert_array_equal(perms,
                                   np.array([[1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 1],
                                             [1, 0, 0],
                                             [1, 0, 0],
                                             [0, 0, 1],
                                             [0, 1, 0]]))

    # ANOVA tests
    def test_f_test_basic1(self):
        N = 9
        mat = np.vstack((
                np.hstack((np.arange(N//3),
                           np.arange(N//3)+100,
                           np.arange(N//3)+200)),
                np.hstack((np.arange(N//3)+100,
                           np.arange(N//3)+300,
                           np.arange(N//3)+400))))
        cats = np.array([0] * (N//3) +
                        [1] * (N//3) +
                        [2] * (N//3),
                        dtype=np.float32)
        nv_f_stats, pvalues = _naive_f_permutation_test(mat, cats)
        perms = _init_categorical_perms(cats)
        np_f_stats, pvalues = _np_k_sample_f_statistic(mat, cats, perms)
        np_test.assert_array_almost_equal(nv_f_stats, np_f_stats, 5)

    def test_f_test_basic2(self):
        N = 9
        mat = np.vstack((
                np.hstack((np.arange(N//3),
                           np.arange(N//3)+100,
                           np.arange(N//3)+200)),
                np.hstack((np.arange(N//3)+100,
                           np.arange(N//3)+300,
                           np.arange(N//3)+400))))
        mat = mat.astype(np.float64)
        cats = np.array([0]*(N//3) +
                        [1]*(N//3) +
                        [2]*(N//3),
                        dtype=np.float32)
        nv_f_stats, pvalues = _naive_f_permutation_test(mat, cats)
        perms = _init_categorical_perms(cats)
        np_f_stats, pvalues = _np_k_sample_f_statistic(mat, cats, perms)
        np_test.assert_array_almost_equal(nv_f_stats, np_f_stats, 5)


if __name__ == '__main__':
    unittest.main()
