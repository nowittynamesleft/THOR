'''
Puts prediction into CAFA format
'''

import sys
import os
from Bio import SeqIO
import Bio

def load_FASTA(filename):
    """ Loads fasta file and returns a list of the Bio SeqIO records """
    infile = open(filename, 'rU')
    entries = [str(entry.id) for entry in SeqIO.parse(infile, 'fasta')]
    if(len(entries) == 0):
        return False
    return entries

def main():
    targets = load_FASTA(sys.argv[1])
    target_to_outfile = dict() 
    outfiles = []
    directory = 'group2_targets'
    for filename in os.listdir(directory):
        curr_targets = load_FASTA(os.path.join(directory, filename))
        taxa_id = filename.split('.')[1]
        outfile = open('cafa_preds/RegionaSpecTHOR_3_' + taxa_id + '.txt', 'w')
        outfiles.append(outfile)
        outfile.write('AUTHOR\tRegionaSpecTHOR\n')
        outfile.write('MODEL\t3\n')
        outfile.write('KEYWORDS\tmachine learning, sequence properties.\n')
        for target in curr_targets:
            target_to_outfile[target] = outfile

    go_terms = open(sys.argv[2], 'r').read().split('\n')
    go_terms.pop()
    preds = open(sys.argv[3], 'r').read().split('\n')
    preds.pop()

    # Now have a list of go terms and a list of targets, can begin putting preds into each CAFA file
    for prediction in preds:
        fields = prediction.split('\t')
        prot = targets[int(fields[0])]
        go_term = go_terms[int(fields[1])]
        val = fields[2]
        if(len(val) < 4): # tack on a zero so it's 1.00 or 0.70 instead of 1.0 or 0.7
            val += '0'
        target_to_outfile[prot].write(prot + '\t' + go_term + '\t' + str(val) + '\n')

    for outfile in outfiles:
        outfile.write('END')
if __name__ == '__main__':
    main()
