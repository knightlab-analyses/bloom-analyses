import numpy as np
import pandas as pd

import unittest
import numpy.testing as np_test

from bloom.stats.permutation import (_init_reciprocal_perms,
                                     _init_categorical_perms,
                                     _naive_mean_permutation_test,
                                     _np_two_sample_mean_statistic)
from bloom.stats.permutation import permutation_mean

from skbio.util._testing import assert_data_frame_almost_equal


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

    def test_permutation_mean(self):
        D = 5
        M = 6
        mat = np.array([range(10)] * M, dtype=np.float32)
        cats = np.array([0] * D + [1] * D, dtype=np.float32)
        permutations = 1000

        table = pd.DataFrame(mat.T)
        grouping = pd.Series(['a']*D+['b']*D)

        nv_stats, nv_p = _naive_mean_permutation_test(mat, cats, permutations)
        res = permutation_mean(table, grouping)
        np_stats, np_p = res.m, res.pvalue
        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        nv_stats = np.array(nv_stats)
        np_stats = np.array(np_stats)
        self.assertEqual(sum(abs(nv_stats-np_stats) > 0.1)[0], [0])

        # Check test statistics
        self.assertAlmostEquals(sum(nv_stats-5), 0, 4)
        self.assertAlmostEquals(sum(np_stats-5), 0, 4)

        # Check for small pvalues
        self.assertEquals(sum(nv_p > 0.05), 0)
        self.assertEquals(sum(np_p > 0.05), 0)

        np_test.assert_array_almost_equal(np.ravel(nv_stats), np_stats)

    def test_permutative_permutation_mean_seed(self):
        D, M = 5, 6
        mat = np.array([range(10)]*M, dtype=np.float32)
        cats = np.array([0]*D+[1]*D, dtype=np.float32)
        permutations = 1000

        table = pd.DataFrame(mat.T)
        grouping = pd.Series(['a']*D+['b']*D)

        nv_stats, nv_p = _naive_mean_permutation_test(mat, cats, permutations)
        res = permutation_mean(table, grouping, random_state=10)
        np_stats, np_p = res.m, res.pvalue
        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        nv_stats = np.array(nv_stats)
        np_stats = np.array(np_stats)
        self.assertEqual(sum(abs(nv_stats-np_stats) > 0.1)[0], [0])

        # Check test statistics
        self.assertAlmostEquals(sum(nv_stats-5), 0, 4)
        self.assertAlmostEquals(sum(np_stats-5), 0, 4)

        # Check for small pvalues
        self.assertEquals(sum(nv_p > 0.05), 0)
        self.assertEquals(sum(np_p > 0.05), 0)
        np_test.assert_array_almost_equal(np.ravel(nv_stats), np_stats)

    def test_permutation_mean_index_m(self):
        table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
                              [9,  11, 12, 10, 10, 10, 10],
                              [1,  11, 10, 11, 10, 5,  9],
                              [22, 21, 9,  10, 10, 10, 10],
                              [20, 22, 10, 10, 13, 10, 10],
                              [23, 21, 14, 10, 10, 10, 10]],
                             index=['s1', 's2', 's3', 's4', 's5', 's6'],
                             columns=['b1', 'b2', 'b3', 'b4',
                                      'b5', 'b6', 'b7'])
        grouping = pd.Series([0, 0, 0, 1, 1, 1],
                             index=['s1', 's2', 's3', 's4', 's5', 's6'])
        results = permutation_mean(table, grouping,
                              permutations=100, random_state=0)
        exp = pd.DataFrame({'m': [14.333333, 10.333333, 0.333333,
                                  0.333333, 1.000000, 1.666667,
                                  0.333333],
                            'pvalue': [0.108910891089, 0.108910891089,
                                       1.0, 1.0, 1.0, 1.0, 1.0]},
                           index=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
        np_test.assert_allclose(results.m.values, exp.m.values, 5)

    @unittest.skip("Randomized seed not configured properly")
    def test_permutation_mean_index(self):
        table = pd.DataFrame([[12, 11, 10, 10, 10, 10, 10],
                              [9,  11, 12, 10, 10, 10, 10],
                              [1,  11, 10, 11, 10, 5,  9],
                              [22, 21, 9,  10, 10, 10, 10],
                              [20, 22, 10, 10, 13, 10, 10],
                              [23, 21, 14, 10, 10, 10, 10]],
                             index=['s1', 's2', 's3', 's4', 's5', 's6'],
                             columns=['b1', 'b2', 'b3', 'b4',
                                      'b5', 'b6', 'b7'])
        grouping = pd.Series([0, 0, 0, 1, 1, 1],
                             index=['s1', 's2', 's3', 's4', 's5', 's6'])
        results = permutation_mean(table, grouping,
                                   permutations=100, random_state=0)
        exp = pd.DataFrame({'m': [14.333333, 10.333333, 0.333333,
                                  0.333333, 1.000000, 1.666667,
                                  0.333333],
                            'pvalue': [0.108910891089, 0.108910891089,
                                       1.0, 1.0, 1.0, 1.0, 1.0]},
                           index=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
        assert_data_frame_almost_equal(results, exp)

    def test_basic_mean1(self):
        # Basic quick test
        np.random.seed(0)
        D = 5
        M = 6
        mat = np.array([range(10)]*M, dtype=np.float32)
        cats = np.array([0]*D+[1]*D, dtype=np.float32)
        permutations = 1000
        perms = _init_reciprocal_perms(cats, permutations)

        nv_stats, nv_p = _naive_mean_permutation_test(mat, cats, permutations)
        np_stats, np_p = _np_two_sample_mean_statistic(mat, perms)

        nv_stats = np.matrix(nv_stats).transpose()
        nv_p = np.matrix(nv_p).transpose()
        nv_stats = np.array(nv_stats)
        np_stats = np.array(np_stats)
        self.assertEqual(sum(abs(nv_stats-np_stats) > 0.1)[0], [0])

        # Check test statistics
        self.assertAlmostEquals(sum(nv_stats-5), 0, 4)
        self.assertAlmostEquals(sum(np_stats-5), 0, 4)

        # Check for small pvalues
        self.assertEquals(sum(nv_p > 0.05), 0)
        self.assertEquals(sum(np_p > 0.05), 0)

        np_test.assert_array_almost_equal(np.ravel(nv_stats), np_stats)

    def test_basic_mean2(self):
        # Basic quick test
        np.random.seed(0)
        D = 5
        M = 6
        mat = np.array([[0]*D+[10]*D]*M, dtype=np.float32)
        cats = np.array([0]*D+[1]*D, dtype=np.float32)
        permutations = 1000
        perms = _init_reciprocal_perms(cats, permutations)

        nv_stats, nv_p = _naive_mean_permutation_test(mat, cats, 1000)
        np_stats, np_p = _np_two_sample_mean_statistic(mat, perms)

        nv_stats = np.array(nv_stats).transpose()
        nv_p = np.array(nv_p).transpose()
        self.assertEquals(sum(abs(nv_stats-np_stats) > 0.1), 0)

        # Check test statistics
        self.assertEquals(int(sum(nv_stats-10)), 0)
        self.assertEquals(int(sum(np_stats-10)), 0)

        # Check for small pvalues
        self.assertEquals(sum(nv_p > 0.05), 0)
        self.assertEquals(sum(np_p > 0.05), 0)

        np_test.assert_array_almost_equal(nv_stats, np_stats)

    def test_large(self):
        # Large test
        N = 10
        mat = np.array(
            np.array(np.vstack((
                np.array([0]*(N//2)+[1]*(N//2)),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.array([0]*N),
                np.random.random(N))), dtype=np.float32))
        cats = np.array([0]*(N//2)+[1]*(N//2), dtype=np.float32)
        permutations = 1000
        perms = _init_reciprocal_perms(cats, permutations)
        np_stats, np_p = _np_two_sample_mean_statistic(mat, perms)

    def test_random_mean(self):
        # Randomized test
        N = 50
        mat = np.array(
            np.array(np.vstack((
                np.array([0]*(N//2)+[100]*(N//2)),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N),
                np.random.random(N))), dtype=np.float32))
        cats = np.array([0]*(N//2)+[1]*(N//2), dtype=np.float32)
        nv_stats, nv_p = _naive_mean_permutation_test(mat, cats, 1000)
        nv_stats = np.array(nv_stats).transpose()
        permutations = 1000
        perms = _init_reciprocal_perms(cats, permutations)
        np_stats, np_p = _np_two_sample_mean_statistic(mat, perms)

        self.assertAlmostEquals(nv_stats[0], 100., 4)
        self.assertAlmostEquals(np_stats[0], 100., 4)

        self.assertLess(nv_p[0], 0.05)
        self.assertLess(np_p[0], 0.05)

        # Check test statistics
        self.assertEquals(sum(nv_stats[1:] > nv_stats[0]), 0)
        self.assertEquals(sum(np_stats[1:] > np_stats[0]), 0)

        np_test.assert_array_almost_equal(np_stats, nv_stats, 4)
