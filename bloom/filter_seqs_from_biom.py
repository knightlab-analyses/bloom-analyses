import biom
import argparse
import sys


def trim_seqs(seqs, seqlength=100):
    """
    Trims the sequences to a given length

    Parameters
    ----------
    seqs: generator of skbio.Sequence objects

    Returns
    -------
    generator of skbio.Sequence objects
        trimmed sequences
    """

    for seq in seqs:

        if len(seq) < seqlength:
            raise ValueError('sequence length is shorter than %d' % seqlength)

        yield seq[:seqlength]

def remove_seqs(table, seqs):
    """
    Parameters
    ----------
    table : biom.Table
       Input biom table
    seqs : generator, skbio.Sequence
       Iterator of sequence objects to be removed from the biom table.

    Return
    ------
    biom.Table
    """
    filter_seqs = {str(s) for s in seqs}
    _filter = lambda v, i, m: i in filter_seqs
    table.filter(_filter, axis='observation', invert=True)
    return table


def main(argv):
    parser=argparse.ArgumentParser(
        description='Filter sequences from biom table using a fasta file. Version ' + \
        __version__)
    parser.add_argument('-i','--inputtable',
                        help='input biom table file name')
    parser.add_argument('-o','--output',
                        help='output biom file name')
    parser.add_argument('-f','--fasta',
                        help='filtering fasta file name')

    args=parser.parse_args(argv)

    seqs = skbio.read(args.fasta)
    table = load_table(args.inputtable)
    length = min(map(len, table.ids(axis='observation')))
    if not parse.ignore_table_seq_length:
        seqs = trim_seqs(seqs, length=length)

    outtable = remove_seqs(table, seqs)

    with biom.util.biom_open(args.output, 'w') as f:
        outtable.to_hdf5(f, "filterbiomseqs")


if __name__ == "__main__":
    main(sys.argv[1:])
