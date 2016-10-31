#!/usr/bin/env python


"""
Filter sequences from a biom table based on a fasta file
used for AG bloom filtering
"""

# amnonscript

__version__ = "1.4"

import biom

import argparse
import sys


def iterfastaseqs(filename):
	"""
	iterate a fasta file and return header,sequence
	input:
	filename - the fasta file name

	output:
	seq - the sequence
	header - the header
	"""

	fl=open(filename,"rU")
	cseq=''
	chead=''
	for cline in fl:
		if cline[0]=='>':
			if chead:
				yield(cseq,chead)
			cseq=''
			chead=cline[1:].rstrip()
		else:
			cseq+=cline.strip()
	if cseq:
		yield(cseq,chead)
	fl.close()


def filterbiombyfasta(tablefilename,outfilename,fastafilename,ignoretablelen=False,number=0):
	"""
	filter sequences from a biom table. The sequences to remove are in a fasta file.

	input:
	tablefilename : str
		name of the input biom table file
	outfilename : str
		name of the output biom table file
	fastafilename : str
		name of the fasta file containing the sequences to remove from the biom table
	ignoretablelen : bool (optional)
		False (default) to trim the fasta sequences to the table sequence length before comparing
		True to not trim
	number : int (optional)
		the number of sequences from the fasta file to use (starting from the beginning) or 0 to use all
	"""
	print('loading biom table %s' % tablefilename)
	table=biom.load_table(tablefilename)
	print('table has %d sequences.' % table.shape[0])
	print('loading fasta file %s' % fastafilename)
	seqset=set()
	numtot=0
	# get the read len
	if not ignoretablelen:
		tseqs=table.ids('observation')
		seqlen=len(tseqs[0])
		for cseq in tseqs:
			if seqlen!=len(cseq):
				ignoretablelen=True
				print('Sequence length not uniform in table! bad sequence is %s' % cseq)
				break

	# iterate over all fasta sequences
	for cseq,chead in iterfastaseqs(fastafilename):
		# if we used enough sequences from the fasta file, we're done
		if number>0:
			if numtot>=number:
				break
		numtot+=1
		# trim the fasta sequence to the table sequence length
		if not ignoretablelen:
			cseq=cseq[:seqlen]
		if table.exists(cseq,axis='observation'):
			seqset.add(cseq)
	print('loaded %d sequences. %d found in table. filtering' % (numtot,len(seqset)))
	# filter removing these sequences in place
	table.filter(list(seqset),axis='observation',invert=True)
	print('%d sequences left. saving' % table.shape[0])
	# save the filtered biom table
	with biom.util.biom_open(outfilename, 'w') as f:
		table.to_hdf5(f, "filterbiomseqs")


def main(argv):
	parser=argparse.ArgumentParser(description='Filter sequences from biom table using a fasta file. Version '+__version__)
	parser.add_argument('-i','--inputtable',help='input biom table file name')
	parser.add_argument('-o','--output',help='output biom file name')
	parser.add_argument('-f','--fasta',help='fitering fasta file name')
	parser.add_argument('-n','--number',help='number of sequences to filter (from start of file) or 0 for all',default=0,type=int)
	parser.add_argument('--ignore-table-seq-length',help="don't trim fasta file sequences to table read length",action='store_true')

	args=parser.parse_args(argv)

	filterbiombyfasta(args.inputtable,args.output,args.fasta,args.ignore_table_seq_length,args.number)

if __name__ == "__main__":
	main(sys.argv[1:])
