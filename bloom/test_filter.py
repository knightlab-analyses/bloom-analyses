from biom import Table
from bloom.filter_seqs_from_biom import remove_seqs, trim_seqs
import skbio
import unittest
import numpy as np
import numpy.testing as npt

class TestFilter(unittest.TestCase):
    def setUp(self):
        self.seqs = (skbio.Sequence('AACCGGTT'),
                     skbio.Sequence('AACCGAGG'),
                     skbio.Sequence('AACCTTTT'),
                     skbio.Sequence('AACCGCTC'))
        self.table = Table(
            np.array([
                [0, 1, 1],
                [0, 2, 1],
                [1, 0, 1],
                [0, 0, 1],
                [9, 1, 1]]),
            ['AACCGG',
             'AACCGA',
             'AACCTT',
             'AACCGC',
             'AAAAAA'],
            ['s1', 's2', 's3'])

    def test_trim_seqs(self):
        seqs = trim_seqs(self.seqs, seqlength=6)
        exp = [skbio.Sequence('AACCGG'),
               skbio.Sequence('AACCGA'),
               skbio.Sequence('AACCTT'),
               skbio.Sequence('AACCGC')]
        self.assertEqual(list(seqs), exp)

    def test_trim_seqs_error(self):
        with self.assertRaises(ValueError):
            list(trim_seqs(self.seqs, seqlength=20))

    def test_remove_seqs(self):
        seqs = trim_seqs(self.seqs, seqlength=6)
        res = remove_seqs(self.table, seqs)
        exp = Table(
            np.array([[9, 1, 1]]),
            ['AAAAAA'],
            ['s1', 's2', 's3'])
        npt.assert_array_equal(np.array(exp.matrix_data.todense()),
                               np.array(res.matrix_data.todense()))


if __name__=='__main__':
    unittest.main()
