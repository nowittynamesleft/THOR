annotation_rnn.py trains a model that takes sequences (in .fasta format) and their functional labels (in a sparse matrix text file, with indices according to the
fasta file). It also takes a list of functions to train and predict on in the text file "function_list.txt". Proteins are represented using protvecs, which are 100-
dimensional representations of trimers.

To handle the dependencies, use a virtual environment like virtualenv and pip install everything that comes up.
Some dependencies:
keras
theano
numpy/scipy
argparse
Biopython
