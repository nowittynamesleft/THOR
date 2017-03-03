'''
Train an RNN on a fasta file of proteins; then saves the models in a folder 
every 50 iterations.
Use generate_seqs.py to generate sequences from a model.
There are 4 models in this file that you can train:
    1. LSTM -> traditional layer
    2. GRU -> traditional layer
    3. GRU -> LSTM -> traditional layer
    4. LSTM -> GRU -> traditional layer
'''
from __future__ import print_function
from keras import backend as K # MUST be before the other keras import statements
K.set_image_dim_ordering('th')
from keras.utils.np_utils import to_categorical
import keras.models
import re
from keras.models import Sequential, Model
from keras.layers import (Input, Embedding, Dense, Activation, Dropout, merge, 
                                        LSTM, GRU, Convolution1D, MaxPooling1D,
                                        Masking, Flatten, Highway, Convolution2D,
                                        MaxPooling2D)
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix
import random
import sys
import Bio
from Bio import SeqIO
import argparse
import os
import collections


def parse_args():
    parser = argparse.ArgumentParser(description=
        'Pick model type and fasta file to train on. For testing: '
        + 'python annotation_rnn.py -modeltype 3 -predname test')
    parser.add_argument('-modeltype', required=True,
                            help='model types 1, 2, 3, or 4')
    parser.add_argument('-fasta', default='group2targets.fasta', 
                            help='fasta file to train on')
    parser.add_argument('-go', default='annotationsCC1701.txt',
                            help='go file for labels to train on')
    parser.add_argument('-iterations', default='10',
                            help='number of iterations to train for')
    parser.add_argument('-loadmodel', 
                    help='model file to load and generate predictions from')
    parser.add_argument('-numseqs', default='-1', help='train on a number of sequences' 
                                    + '(-1 for all sequences in uniprot file)')
    parser.add_argument('-maxlen', default='500',
            help='train on only sequences less than' 
                                    + ' maxlen amino acids')
    parser.add_argument('-predname', required=True, help='name of prediction file')
    parser.add_argument('-func_file', default='function_list.txt', help='file that contains which functions to train on')
    parser.add_argument('-fake', default='false', help='true if you want to generate fake data')
    parser.add_argument('-protvecs', default='protVec_100d_3grams.csv', help='filename of protvecs csv')
    return parser.parse_args()


def load_FASTA(filename):
    """ Loads fasta file and returns a list of the Bio SeqIO records """
    infile = open(filename, 'rU')
    entries = [str(entry.seq) for entry in SeqIO.parse(infile, 'fasta')]
    if(len(entries) == 0):
        return False
    return entries


def make_prot_annotation_matrix(annotation_file):
    '''
    Makes dictionary of annotations with the index of proteins as the key
    and the list of the function indices as the value
    '''
    sparse_annotations = np.loadtxt(annotation_file)
    rows = sparse_annotations[:, 0]
    columns = sparse_annotations[:, 1]
    vals = sparse_annotations[:, 2]
    annotation_matrix = coo_matrix((vals, (rows, columns))).todense()
    print(annotation_matrix)
    return annotation_matrix


def function_count(annotations):
    '''
    Returns the maximally indexed function it has come across
    '''
    functions = set()
    for prot in annotations:
        functions = functions.union(annotations[prot])
    return max(functions)


def remove_long_prots(entries, maxlen):
    tokenized_sequences = []
    removed_entries = []
    # Remove proteins that are too long
    entry_idx = 0
    removed = 0
    for entry_idx in range(0, len(entries)):
        entry = entries[entry_idx]
        if(len(entry) > maxlen):
            removed += 1
            entry_chars = []
            removed_entries.append(entry_idx)
        else:
            entry_chars = list(entry)
            entry_chars = [char for char in entry_chars if char != '\n']
        tokenized_sequences.append(entry_chars)
    print('Too long: ' + str(removed))
    print('Number of proteins found: ' + str(len(tokenized_sequences)))
    for idx in sorted(removed_entries, reverse=True):
        del tokenized_sequences[idx]
    return tokenized_sequences, removed_entries


def remove_prots(entries, maxlen, annotations):
    tokenized_sequences = []
    removed_entries = []
    # Remove proteins that are too long
    entry_idx = 0
    removed = 0
    for entry_idx in range(0, len(entries)):
        entry = entries[entry_idx]
        if(len(entry) > maxlen or annotations[entry_idx] == 0):
            removed += 1
            entry_chars = []
            removed_entries.append(entry_idx)
        else:
            entry_chars = list(entry)
            entry_chars = [char for char in entry_chars if char != '\n']
        tokenized_sequences.append(entry_chars)
    print('Too long/neutral example: ' + str(removed))
    return tokenized_sequences, removed_entries


def get_removed_funcs(annotations, removed_entries):
    # Need to count the number of instances of each function are in the
    # considered proteins
    functions = {}
    for prot in annotations:
        if prot not in removed_entries:
            for function in annotations[prot]:
                if(function in functions):
                    functions[function] += 1
                else:
                    functions[function] = 1

    # Select which functions will be removed from data
    removed_funcs = []
    for function in functions:
        if functions[function] not in range(100, 100000):
            # if it's not in the range remove it
            removed_funcs.append(function)
    print('Number of functions before removing: ' + str(len(functions)))
    print('Removed functions: ' + str(len(removed_funcs)))
    function_list = sorted(functions.keys())
    return removed_funcs, function_list


def consecutive_residues(sequence):
    seq = ''.join([c for c in sequence if c != '0.0'])
    if re.search(r'(.)\1\1', seq):
	return 1
    else:
        return 0
   

def get_fake_data(entries, maxlen, uniprot_filename, predname, selection,
                  num_seqs):
    tokenized_sequences, removed_entries = remove_long_prots(entries, maxlen)
    chars, char_indices, indices_char = get_char_indices(entries)
    old_to_new_prot_inds, new_to_old_prot_inds = index_conversions(
                                range(0, len(tokenized_sequences)),
                                removed_entries)
    X = get_seq_vecs(removed_entries, tokenized_sequences, uniprot_filename,
                        selection, maxlen, char_indices)
    y = np.zeros((X.shape[0]), dtype=np.bool)
    idx = 0
    for sequence in X:
        if consecutive_residues(tokenized_sequences[idx]):
            y[idx] = 1
        else:
            y[idx] = 0
        idx += 1
    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.25)
    print('y.shape:')
    print(y.shape)
    print('fraction of positive examples:')
    print(np.sum(y>0)/float(len(y)))
    return [X_train[:num_seqs], X_test, y_train[:num_seqs], y_test, 
            char_indices, indices_char, chars,
            old_to_new_prot_inds, new_to_old_prot_inds]

def curr_func_ratio(raw_annotations, func):
    frac = float(sum(raw_annotations[:, func] == 1.0))/float(sum(raw_annotations[:, func] != 0))
    print("Function: " + str(func) + " Pos ex ratio: " + str(frac))
    return frac 

def get_one_func_data(X_to_predict, raw_annotations, func):

    print('Function index: ' + str(func))
    print('raw_annotations shape: ' + str(raw_annotations.shape))
    annots = raw_annotations[:, func]
    mask = np.array(annots != 0)
    mask = np.reshape(np.array(annots != 0), (mask.shape[0]))
    annotations = annots[mask] # get only those annotations that aren't 0
    print('Mask shape before')
    print(mask.shape)
    print('X_to_predict.shape: ' + str(X_to_predict.shape))
    X = X_to_predict[mask]
    print('annotations vector shape:')
    print(annotations.shape)
    print('X.shape')
    print(X.shape)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    print('X.shape after reshape')
    print(X.shape)
    y = np.zeros((X.shape[0]), dtype=np.float)
    print('y.shape:')
    print(y.shape)

    for i in range(0, X.shape[0]):
        if(annotations[i] > 0):
            y[i] = 1
        else:
            y[i] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.25)

    print('fraction of positive examples:')
    print(np.sum(y>0)/float(len(y)))
    return [X_train, X_test, y_train, y_test]


def get_seq_vecs(removed_entries, tokenized_sequences, uniprot_filename,
                    selection, maxlen, char_indices):
    expected_seq_vec_filename = ('./predictions/seq_vectors_' 
            + uniprot_filename + 'model_' + str(selection) + '_maxlen_' 
            + str(maxlen) + '_fake' + '.npy')
    print('New number of tokenized sequences after removal of '
            + str(len(removed_entries)) + ' sequences: '
            + str(len(tokenized_sequences)))
    print('Generating seq vectors')
    sequences = center_sequences(tokenized_sequences, maxlen=maxlen,
                                                    dtype=np.str_)
    one_hot_seqs = np.zeros((len(sequences), maxlen, len(char_indices)))
    for i in range(0, len(sequences)):
        for j in range(0, len(sequences[i])):
            one_hot_seqs[i][j][char_indices[sequences[i][j]]] = 1

    return one_hot_seqs


def make_protvec_dict(protvec_filename):
    trimer_to_protvec = {}
    lines = open(protvec_filename, 'rU').read().split('\n')
    for line in lines:
        lin = line[1:-1]
        fields = lin.split('\t')
        trimer = fields[0]
        protvec = np.array(fields[1:])
        trimer_to_protvec[trimer] = protvec

    return trimer_to_protvec


def get_protvecs(removed_entries, tokenized_sequences, trimer_to_protvec, maxlen):
    # Using the prot_vec_dict, transform the tokenized sequences into vectors
    # Create three trimer sets for each sequence
    new_seqs = tokenized_sequences

    print('Length of new_seqs: ' + str(len(new_seqs)))
    X = np.zeros((len(new_seqs), 1, 300, maxlen - 5)) #shape of trimer vectors, 300 dimensions (100 for each vector in the stride) x maxlen x numseqs)
    print('X.shape initially: ' + str(X.shape))
    for seq_num, sequence in enumerate(new_seqs):
        for i in range(0,len(sequence) - 5): # - 2 for trimer, -3 for three trimers per iteration
            for part in range(0, 3):
                trimer = ''.join(sequence[i + part:i + part + 3])
                if trimer in trimer_to_protvec:
                    X[seq_num, 0, part*100:(part + 1)*100, i] = trimer_to_protvec[trimer]
                else:
                    X[seq_num, 0, part*100:(part + 1)*100, i] = trimer_to_protvec['<unk>']
    return X

def center_sequences(sequences, maxlen, value=0.0, dtype=np.float):
    nb_samples = len(sequences)
    x = (np.zeros((nb_samples, maxlen))).astype(dtype)
    print(x.shape)
    for idx, s in enumerate(sequences):
        trunc = np.asarray(s, dtype=dtype)
        if(maxlen != len(trunc)):
            x[idx, (maxlen - len(trunc))/2:-(maxlen - len(trunc))/2] = trunc
        else:
            x[idx] = trunc
    return x


def get_char_indices(entries):
    text = ''.join(entries)
    chars = sorted(list(set(text)))
    chars.insert(0, '0.0')
    print(chars)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return chars, char_indices, indices_char


def get_sequence_from_vector(indices_char, vector_sequence):
    seqs_X_train = np.copy(vector_sequence)
    for i, c in indices_char.iteritems():
        seqs_X_train[vector_sequence == str(i)] = c


def index_conversions(inds, removed_inds):
    # inds should be sorted in increasing order
    # Also, they can have gaps i.e. 0, 4, 5
    old_to_new_inds = {}
    count = 0
    for i in range(0, len(inds)): 
        if inds[i] in removed_inds:
            count += 1
        else:
            old_to_new_inds[inds[i]] = i - count
    new_to_old_inds = dict((v,k) for k,v in old_to_new_inds.iteritems())
    return old_to_new_inds, new_to_old_inds


def main(args):
    entries = load_FASTA(args.fasta)
    num_seqs = int(args.numseqs)
    annotations = make_prot_annotation_matrix(args.go)
    maxlen = int(args.maxlen)
    if(maxlen == -1):
        longest_prot = max(entries, key=len)
        maxlen = len(longest_prot)

    print("Max length of protein: " + str(maxlen))
    chars, char_indices, indices_char = get_char_indices(entries)
    tokenized_sequences, removed_entries = remove_long_prots(entries, maxlen)
    #Remove from annotations
    print('Length of tokenized sequences: ' + str(len(tokenized_sequences)))
    print('annotations shape before')
    print(annotations.shape)
    mask = np.ones(annotations.shape[0], dtype=bool)
    mask[removed_entries] = False
    annotations = annotations[mask]
    old_to_new_prot_inds, new_to_old_prot_inds = index_conversions(
                                range(0, len(tokenized_sequences)),
                                removed_entries) # just for index conversions after predictions are made
    if args.protvecs.lower() == 'false':
        expected_seq_vec_filename = './predictions/one_hot_maxlen_' + str(maxlen) + '.npy'
        if os.path.isfile(expected_seq_vec_filename + 'lol'):
            print("Loading one hot numpy mat")
            X_to_predict = np.load(expected_seq_vec_filename)
        else:
            print("Generating one hot vecs for fasta")
            X_to_predict = get_seq_vecs(removed_entries, tokenized_sequences, args.fasta,
                       args.modeltype, maxlen, char_indices)
            np.save(expected_seq_vec_filename, X_to_predict)
    else:
        trimer_to_protvec = make_protvec_dict(args.protvecs)
        expected_seq_vec_filename = './predictions/ProtVec_maxlen_' + str(maxlen) + '.npy'
        if os.path.isfile(expected_seq_vec_filename + 'lol'):
            print("Loading protvecs numpy mat")
            X_to_predict = np.load(expected_seq_vec_filename)
        else:
            print("Generating protvecs for fasta")
            X_to_predict = get_protvecs(removed_entries, tokenized_sequences, trimer_to_protvec, maxlen)
            np.save(expected_seq_vec_filename, X_to_predict)
    print('X_to_predict shape')
    print(X_to_predict.shape)
    print('annotations shape')
    print(annotations.shape)
    
    list_of_funcs = open(args.func_file, 'r').read().split('\n')
    list_of_funcs.pop()

    prediction_matrix = np.zeros_like(annotations)
    for func in list_of_funcs:
        if(args.fake.lower() == 'true'):
            print('Getting fake data')
            [X, X_test, y, y_test, char_indices, indices_char, chars,
                    _, _] = get_fake_data(entries, maxlen,
                            args.fasta, args.predname, args.modeltype,
                            num_seqs)
        else: 
            print('Getting data from annotation file')
            if not curr_func_ratio(annotations, int(func)) > 0.1:
                continue
            else:
                [X, X_test, y, y_test] = get_one_func_data(X_to_predict, annotations, int(func))
        new_to_old_func_inds = {0:0}
        print('X')
        print(X.shape)
        print('y')
        print(y.shape)
        print(collections.Counter(y))
        frac = float(collections.Counter(y)[1.0])/float(sum(collections.Counter(y).values()))
        
        model = build_model(int(args.modeltype), maxlen, len(new_to_old_func_inds), len(indices_char), X)
        if model == None:
            print("Not a valid selection of model. Stopping.")
            return
        elif(int(args.modeltype) < 4):

            optimizer = RMSprop(lr=0.01)
            #optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9)
            model.compile(loss='binary_crossentropy', optimizer=optimizer,
                                                         metrics=['accuracy'])
            if args.loadmodel != None:
                print('Loading model...')
                model.load_weights(args.loadmodel)
                preds = make_predictions(model, X_to_predict, new_to_old_prot_inds, new_to_old_func_inds, len(entries))
                prediction_matrix[:, func] = preds[:, 0]
            else:
                print('Training model...')
                train_model(args.iterations, model, X, y, args.modeltype,
                                    len(entries), args.predname)
                print('Prediction after training. Shape of X_to_predict: ' + str(X_to_predict.shape)) 
                preds = make_predictions(model, X_to_predict, new_to_old_prot_inds, new_to_old_func_inds,
                    len(entries))
                prediction_matrix[:, func] = preds[:, 0]
            if(args.loadmodel == None and num_seqs != -1):
                print('\n' + str(test_model(model, X_test, y_test)))
            #test_model(model, X_test, y_test)
        elif(int(args.modeltype) == 4):
            print('Fitting random forest...')
            model.fit(X,y)
            print('Predicting test data')
            y_predicted = model.predict(X_test)
            print('Calculating accuracy: ')
            print(accuracy_score(y_test, y_predicted, normalize=True))
   
    print('Saving numpy prediction matrix')
    np.save('prediction_matrix' + args.predname + '.npy', prediction_matrix)
    print('Done')
    # Write prediction matrix to text file
    print('Writing prediction matrix to text file')
    outfile = open('predictionsgroup2ccALL.txt', 'w')
    for i in range(prediction_matrix.shape[0]):
        for j in range(prediction_matrix.shape[1]):
            if prediction_matrix[i,j] != 0.00:
                outfile.write(str(i) + '\t' + str(j) + '\t' + str(prediction_matrix[i,j]) + '\n')

def test_model(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    cmat = confusion_matrix(y_test, y_pred)
    per_class_acc = cmat.diagonal().astype('float')/cmat.sum(axis=1).astype('float')
    print(per_class_acc)
    print(cmat.diagonal())
    print(cmat.sum(axis=1))
    print(model.metrics_names)
    print(model.evaluate(X_test, y_test, batch_size=64, verbose=1, 
                                                            sample_weight=None))


def build_model(selection, maxlen, output_size, input_alphabet_size, X):
    model = None
    if(selection == 1):
        print('Build model 1: LSTM, dense, activation')
        model = Sequential()
        '''
        model.add(Convolution1D(20, 10, input_dim=input_alphabet_size,
            input_length=maxlen))
        '''
        model.add(LSTM(25, return_sequences=True, input_dim=input_alphabet_size, 
            input_length=maxlen, activation='softsign'))
        model.add(LSTM(25, activation='softsign'))
        model.add(Dense(output_size, activation='sigmoid'))
        #model.summary()

    elif(selection == 2):
        print('Build model 2: conv1d instead of LSTM')
        model = Sequential()
        model.add(Convolution1D(32, 3, input_dim=input_alphabet_size, 
            input_length=maxlen))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_length=3))

        model.add(Convolution1D(16, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_length=3))

        model.add(Convolution1D(32, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_length=3))

        model.add(Flatten())
        model.add(Dense(output_size))
        model.add(Activation('sigmoid'))
        #model.summary()

    elif(selection == 3):
        print('Build model 2: conv2d')
        model = Sequential()
        print(X.shape[1])
        print(X.shape[2])
        print(X.shape[3])
        model.add(Convolution2D(32, 9, 9, input_shape=(X.shape[1], X.shape[2], X.shape[3])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Convolution2D(16, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Flatten())
        model.add(Dense(output_size))
        model.add(Activation('sigmoid'))
    
    elif(selection == 4):
        print('Build model 4: Random Forest Classifier')
        model = RandomForestClassifier()

    return model


def train_model(iterations, model, X, y, selection, num_seqs, predname):
    # train the model, output generated text after each iteration
    print(np.sum(y==1)/float(len(y)))
    pos = np.sum(y==1)
    neg = len(y) - pos
    prop_pos = float(pos)/float(len(y))
    prop_neg = float(neg)/float(len(y))
    print('Prop pos: ' + str(prop_pos))
    print('Prop neg: ' + str(prop_neg))
    for iteration in range(1, int(iterations) + 1):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        
        model.fit(X, y, batch_size=64, nb_epoch=1, class_weight={0:prop_pos, 1:prop_neg}) # opposite proportions
        model_filename = ('models/Selection_' + str(selection) + '.Iteration' 
                                                    + str(iteration) + '.h5')
        print('Saving weights in ' + model_filename)
        model.save_weights(model_filename)


def make_predictions(model, X, new_to_old_prot_inds, new_to_old_func_inds, num_seqs):
    print('Length of new to old prot inds: ' + str(len(new_to_old_prot_inds)))
    print('Making predictions...')
    predictions = model.predict(X)
    print(predictions[:10])
    max_pred = np.amax(predictions, (0,1))
    print('Max prediction: ' + str(max_pred))
    min_pred = np.amin(predictions, (0,1))
    print('Min prediction: ' + str(min_pred))
    mean_pred = np.mean(predictions, axis=0)
    print('Average prediction: ' + str(mean_pred))
    prediction_column = np.zeros((num_seqs, 1), dtype=float)
    print('Column and then predictions')
    print(prediction_column.shape)
    print(predictions.shape)
    for i in range(0, len(predictions)):
        prediction_column[new_to_old_prot_inds[i]] = round(predictions[i], 2)

    return prediction_column


if __name__ == '__main__':
    main(parse_args())
