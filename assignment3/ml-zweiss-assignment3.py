__author__ = 'zweiss'

# =====================================================================================================================
# Homework assignment 3
# HS Machine Learning for Computational Linguists
# SS 2016
# Zarah L. Weiß
# =====================================================================================================================

import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Activation, Reshape
from nltk import ngrams
from collections import Counter


# =====================================================================================================================
# Read data
# =====================================================================================================================

def read_all_files(cur_dir):

    # get a listing of all plain txt files in the dir and all sub dirs
    txt_file_list = []
    for root, dirs, files in os.walk(cur_dir):
        for name in files:
            if name.endswith('.txt'):
                txt_file_list.append(os.path.join(root, name))

    # read all files and save them to list
    txt_content_list = []
    for txt_file in txt_file_list:
        txt_content_list.append(read_file(txt_file))

    return txt_file_list, txt_content_list


def read_file(input_file):

    # get file content
    with open(input_file, 'r') as input_stream:
        text = input_stream.read()
    input_stream.close()

    return text


# =====================================================================================================================
# Task 2: Cargi's methods
# =====================================================================================================================

def mlp_model(input_dimensions, n_out_dim, activation_function):
    """ This function creates two-layer networks
    """
    # start sequential neural network
    m = Sequential()

    # input layer with varying parameters
    m.add(Dense(input_dim=input_dimensions, output_dim=n_out_dim, activation=activation_function))

    # hidden layer
    #m.add(Dense(output_dim=n_out_dim, activation='relu'))
    m.add(Dense(output_dim=n_out_dim, activation='relu'))

    # l2 regularisation
    # dropout
    # or "hidden layer" weg

    # single binary output to 1 or 0
    m.add(Dense(output_dim=1, activation='sigmoid'))

    # compile model getting accuracy
    m.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return m


def cnn_model(document_matrix):

    cnn = Sequential()

    # apply a 2 dimensional 2x50, using 64 output filters on a 2000x50 document matrix
    # border_mode may be same or valid
    cnn.add(Convolution2D(nb_filter=10, nb_row=3, nb_col=document_matrix.shape[2], border_mode='same',
                          input_shape=(1, document_matrix.shape[1], document_matrix.shape[2])))
    cnn.add(Activation('relu'))

    # add another 2 dimensional convolution with only 32 output filters
    cnn.add(Convolution2D(nb_filter=5, nb_row=2, nb_col=document_matrix.shape[2], border_mode='same'))
    cnn.add(Activation('relu'))

    # do pooling
    # this pool size halves the dimensions,
    cnn.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th'))

    # reshape
    # cnn.add(Reshape(32*int(document_matrix.shape[1]/2)*int(document_matrix.shape[2]/2)))
    cnn.add(Reshape((5*int(document_matrix.shape[1]/2)*int(document_matrix.shape[2]/2),)))

    # add fully connected layer
    cnn.add(Dense(output_dim=5, activation='relu'))

    # end
    cnn.add(Dense(output_dim=1, activation='sigmoid'))
    # no sgd for cnns, takes too long
    cnn.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return cnn


def train_test(model, x, y, nfolds):
    """ This function trains and tests a Keras model with k-fold cv.
        'folds' is the array returned by sklearn *KFold splits.
    """
    acc_sum = 0
    for trn_i, test_i in nfolds:
        model.fit(x[trn_i], y[trn_i], nb_epoch=20)
        _, acc = model.evaluate(x[test_i], y[test_i])
        acc_sum += acc
    return acc_sum/len(nfolds)


def train_test_cnn_model(x, y, nfolds):
    x_reshaped = x.reshape(-1, 1, x.shape[1], x.shape[2])
    acc_sum = 0
    summaries = []
    for trn_i, test_i in nfolds:
        cnn = cnn_model(x)
        cnn.fit(x_reshaped[trn_i], y[trn_i], nb_epoch=20)
        _, acc = cnn.evaluate(x_reshaped[test_i], y[test_i])
        acc_sum += acc
        summaries.append(cnn.summary())
    return acc_sum/len(nfolds), summaries




# =====================================================================================================================
# Main
# =====================================================================================================================

if __name__ == '__main__':

    # =================================================================================================================
    # 0. obtain data
    # =================================================================================================================

    # 0.a   read files

    # fix random seed for reproducibility
    seed = 4050712
    np.random.seed(seed)

    dir_pos = '/Users/zweiss/Documents/Uni/9_SS16/iscl-s9_ss16-machine_learning/assignment3/data/review_polarity/txt_sentoken/pos/'
    dir_neg = '/Users/zweiss/Documents/Uni/9_SS16/iscl-s9_ss16-machine_learning/assignment3/data/review_polarity/txt_sentoken/neg/'

    # read text data
    _, reviews_pos = read_all_files(dir_pos)
    _, reviews_neg = read_all_files(dir_neg)
    review_texts = reviews_pos + reviews_neg

    # make labels for prediction
    classes = [True]*len(reviews_pos) + [False]*len(reviews_neg)

    # http://keras.io/getting-started/sequential-model-guide/

    # get ngrams
    ngram_count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    ngram_counts = ngram_count_vectorizer.fit_transform(review_texts)
    ngram_counts.shape

    # =================================================================================================================
    # Task 1
    # Train an evaluate a logistic regression classifier, based on both word unigrams, and word bigrams. (You can limit
    # the features to the words that appear at least N (e.g., 5) times in the corpus.) Report average accuracy of your
    # models.
    # =================================================================================================================

    folds = StratifiedKFold(classes, 10)  # creates an index to be used in cross validation
    log_reg = LogisticRegression()
    log_reg_accuracy = cross_val_score(log_reg, ngram_counts, classes, cv=10, scoring='accuracy')
    print(log_reg_accuracy.mean())
    # 0.8535

    # =================================================================================================================
    # Task 2
    # Repeat the first exercise, but this time use word vectors, with a multi-layer perceptron. You can represent each
    # document as the sum (or average) of all the word vectors in the document (for bigrams you can concatenate the
    # vectors).
    # Report average accuracy.
    # =================================================================================================================

    # get word vectors for words contained in reviews
    unigram_embeddings = {}
    glove_dir = "./rsrc/glove.6B/glove.6B.50d.txt"
    n_dim = 50  # save dimensions of word vector
    glove_file = open(glove_dir, 'r')
    for line in glove_file.readlines():  # iterate through entries
        entry = line.split()
        # print('key: ' + str(entry[0]))
        # print('vec: ' + str(entry[1:]))
        unigram_embeddings[entry[0]] = np.array([float(value) for value in entry[1:]])
    glove_file.close()

    # =================================================================================================================
    # 2.1. use unigrams
    # =================================================================================================================

    # create unigram features
    # unigram_count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    # unigram_counts = unigram_count_vectorizer.fit_transform(review_texts)
    # unigram_counts.shape
    doc_matrix_averaged_unigrams = np.zeros((len(review_texts), n_dim))
    for i_doc in range(0, len(review_texts)):
        tokens = review_texts[i_doc].split()
        unigrams = Counter(ngrams(tokens, 1))
        for unigram in unigrams:
            if unigram[0] in unigram_embeddings.keys():
                doc_matrix_averaged_unigrams[i_doc] += unigram_embeddings[unigram[0]] * unigrams[unigram]  # sup up
            else:
                continue
        doc_matrix_averaged_unigrams[i_doc] /= len(unigrams)  # average
    doc_matrix_averaged_unigrams.shape

    # build a multi layer perceptrons using variyng parameters

    # first: very simple model
    mlp_basic = mlp_model(doc_matrix_averaged_unigrams.shape[1], 50, 'relu')
    mlp_basic_accuracy = train_test(mlp_basic, doc_matrix_averaged_unigrams, np.array(classes), folds)
    mlp_basic_accuracy
    # 0.6465
    mlp_basic.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_1 (Dense)                  (None, 50)            2550        dense_input_1[0][0]
    # ____________________________________________________________________________________________________
    # dense_2 (Dense)                  (None, 50)            2550        dense_1[0][0]
    # ____________________________________________________________________________________________________
    # dense_3 (Dense)                  (None, 1)             51          dense_2[0][0]
    # ====================================================================================================
    # Total params: 5151
    # ____________________________________________________________________________________________________

    # second: increase number of input layers
    mlp_1 = mlp_model(doc_matrix_averaged_unigrams.shape[1], 100, 'relu')
    mlp_1_accuracy = train_test(mlp_1, doc_matrix_averaged_unigrams, np.array(classes), folds)
    mlp_1_accuracy
    # 0.6600, i.e. slowly going somewhere
    mlp_1.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_4 (Dense)                  (None, 100)           5100        dense_input_2[0][0]
    # ____________________________________________________________________________________________________
    # dense_5 (Dense)                  (None, 100)           10100       dense_4[0][0]
    # ____________________________________________________________________________________________________
    # dense_6 (Dense)                  (None, 1)             101         dense_5[0][0]
    # ====================================================================================================
    # Total params: 15301
    # ____________________________________________________________________________________________________

    # second: increase number of input layers
    mlp_2 = mlp_model(doc_matrix_averaged_unigrams.shape[1], 150, 'relu')
    mlp_2_accuracy = train_test(mlp_2, doc_matrix_averaged_unigrams, np.array(classes), folds)
    mlp_2_accuracy
    # 0.6760, i.e. better
    mlp_2.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_10 (Dense)                 (None, 150)           7650        dense_input_4[0][0]
    # ____________________________________________________________________________________________________
    # dense_11 (Dense)                 (None, 150)           22650       dense_10[0][0]
    # ____________________________________________________________________________________________________
    # dense_12 (Dense)                 (None, 1)             151         dense_11[0][0]
    # ====================================================================================================
    # Total params: 30451
    # ____________________________________________________________________________________________________

    # third: increase number of input layers
    mlp_3 = mlp_model(doc_matrix_averaged_unigrams.shape[1], 151, 'relu')
    mlp_3_accuracy = train_test(mlp_3, doc_matrix_averaged_unigrams, np.array(classes), folds)
    mlp_3_accuracy
    # 0.6975, i.e. even better
    mlp_3.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_25 (Dense)                 (None, 151)           7701        dense_input_9[0][0]
    # ____________________________________________________________________________________________________
    # dense_26 (Dense)                 (None, 151)           22952       dense_25[0][0]
    # ____________________________________________________________________________________________________
    # dense_27 (Dense)                 (None, 1)             152         dense_26[0][0]
    # ====================================================================================================
    # Total params: 30805
    # ____________________________________________________________________________________________________

    # fourth: increase number of input layers
    mlp_4 = mlp_model(doc_matrix_averaged_unigrams.shape[1], 152, 'relu')
    mlp_4_accuracy = train_test(mlp_4, doc_matrix_averaged_unigrams, np.array(classes), folds)
    mlp_4_accuracy
    # 0.6660, i.e. worse
    mlp_4.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_22 (Dense)                 (None, 152)           7752        dense_input_8[0][0]
    # ____________________________________________________________________________________________________
    # dense_23 (Dense)                 (None, 152)           23256       dense_22[0][0]
    # ____________________________________________________________________________________________________
    # dense_24 (Dense)                 (None, 1)             153         dense_23[0][0]
    # ====================================================================================================
    # Total params: 31161
    # ____________________________________________________________________________________________________

    # =================================================================================================================
    # 2.2. use bigrams
    # =================================================================================================================

    # create bigram features
    # bigram_count_vectorizer = CountVectorizer(ngram_range=(2, 2))
    # bigram_counts = bigram_count_vectorizer.fit_transform(review_texts)
    # bigram_counts.shape
    doc_matrix_averaged_bigrams = np.zeros((len(review_texts), (n_dim*2)))
    doc_matrix_averaged_bigrams.shape
    for i_doc in range(0, len(review_texts)):
        tokens = review_texts[i_doc].split()
        bigrams = Counter(ngrams(tokens, 2))
        for bigram in bigrams:
            bi1 = unigram_embeddings[bigram[0]] if bigram[0] in unigram_embeddings.keys() else np.zeros(50)
            bi2 = unigram_embeddings[bigram[1]] if bigram[1] in unigram_embeddings.keys() else np.zeros(50)
            doc_matrix_averaged_bigrams[i_doc] += np.concatenate((bi1, bi2), axis=0) * bigrams[bigram]  # sum up
        doc_matrix_averaged_bigrams[i_doc] /= len(bigrams)  # average
    doc_matrix_averaged_bigrams.shape

    # build a multi layer perceptrons using variyng parameters

    # first: very simple model
    mlp_basic_bi = mlp_model(doc_matrix_averaged_bigrams.shape[1], 50, 'relu')
    mlp_basic_bi_accuracy = train_test(mlp_basic_bi, doc_matrix_averaged_bigrams, np.array(classes), folds)
    mlp_basic_bi_accuracy
    # 0.6445, i.e. worse than unigrams
    mlp_basic_bi.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_28 (Dense)                 (None, 50)            5050        dense_input_10[0][0]
    # ____________________________________________________________________________________________________
    # dense_29 (Dense)                 (None, 50)            2550        dense_28[0][0]
    # ____________________________________________________________________________________________________
    # dense_30 (Dense)                 (None, 1)             51          dense_29[0][0]
    # ====================================================================================================
    # Total params: 7651
    # ____________________________________________________________________________________________________

    # increase number of input layers
    mlp_bi_1 = mlp_model(doc_matrix_averaged_bigrams.shape[1], 100, 'relu')
    mlp_bi_1_accuracy = train_test(mlp_bi_1, doc_matrix_averaged_bigrams, np.array(classes), folds)
    mlp_bi_1_accuracy
    # 0.6530, i.e. better
    mlp_bi_1.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_31 (Dense)                 (None, 100)           10100       dense_input_11[0][0]
    # ____________________________________________________________________________________________________
    # dense_32 (Dense)                 (None, 100)           10100       dense_31[0][0]
    # ____________________________________________________________________________________________________
    # dense_33 (Dense)                 (None, 1)             101         dense_32[0][0]
    # ====================================================================================================
    # Total params: 20301
    # ____________________________________________________________________________________________________

    # increase number of input layers
    mlp_bi_2 = mlp_model(doc_matrix_averaged_bigrams.shape[1], 200, 'relu')
    mlp_bi_2_accuracy = train_test(mlp_bi_2, doc_matrix_averaged_bigrams, np.array(classes), folds)
    mlp_bi_2_accuracy
    # 0.6620, i.e. better
    mlp_bi_2.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_37 (Dense)                 (None, 200)           20200       dense_input_13[0][0]
    # ____________________________________________________________________________________________________
    # dense_38 (Dense)                 (None, 200)           40200       dense_37[0][0]
    # ____________________________________________________________________________________________________
    # dense_39 (Dense)                 (None, 1)             201         dense_38[0][0]
    # ====================================================================================================
    # Total params: 60601
    # ____________________________________________________________________________________________________

    # increase number of input layers
    mlp_bi_3 = mlp_model(doc_matrix_averaged_bigrams.shape[1], 210, 'relu')
    mlp_bi_3_accuracy = train_test(mlp_bi_3, doc_matrix_averaged_bigrams, np.array(classes), folds)
    mlp_bi_3_accuracy
    # 0.6720, i.e. better
    mlp_bi_3.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_49 (Dense)                 (None, 210)           21210       dense_input_17[0][0]
    # ____________________________________________________________________________________________________
    # dense_50 (Dense)                 (None, 210)           44310       dense_49[0][0]
    # ____________________________________________________________________________________________________
    # dense_51 (Dense)                 (None, 1)             211         dense_50[0][0]
    # ====================================================================================================
    # Total params: 65731
    # ____________________________________________________________________________________________________

    # increase number of input layers
    mlp_bi_4 = mlp_model(doc_matrix_averaged_bigrams.shape[1], 211, 'relu')
    mlp_bi_4_accuracy = train_test(mlp_bi_4, doc_matrix_averaged_bigrams, np.array(classes), folds)
    mlp_bi_4_accuracy
    # 0.6625, i.e. worse
    mlp_bi_4.summary()
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # dense_46 (Dense)                 (None, 211)           21311       dense_input_16[0][0]
    # ____________________________________________________________________________________________________
    # dense_47 (Dense)                 (None, 211)           44732       dense_46[0][0]
    # ____________________________________________________________________________________________________
    # dense_48 (Dense)                 (None, 1)             212         dense_47[0][0]
    # ====================================================================================================
    # Total params: 66255
    # ____________________________________________________________________________________________________

    # =================================================================================================================
    # Task 3
    # Design and train a convolutional neural network (CNN) for the same task. You are free to choose the architecture,
    # but make sure your convolutions cover at least bigrams, and the model learns multiple convolutions (features
    # maps).
    # Briefly explain your network architecture, and report the accuracy of the model.
    # =================================================================================================================

    # create 3 dimensional matrix: documents x tokens x word dimensions
    token_limit = 200  # set to 200
    doc_matrix_3d = np.zeros((len(review_texts), token_limit, n_dim))
    for i_doc in range(0, len(review_texts)):
        tokens = review_texts[i_doc].split()
        for i_token in range(0, min(len(tokens), token_limit)):
            if tokens[i_token] in unigram_embeddings.keys():
                doc_matrix_3d[i_doc][i_token] = unigram_embeddings[tokens[i_token]]
            else:
                continue
    doc_matrix_3d.shape

    cnn_accuracy, _ = train_test_cnn_model(doc_matrix_3d, np.array(classes), folds)
    cnn_accuracy
    # 0.5225
    # ____________________________________________________________________________________________________
    # Layer (type)                     Output Shape          Param #     Connected to
    # ====================================================================================================
    # convolution2d_25 (Convolution2D) (None, 10, 200, 50)   1510        convolution2d_input_13[0][0]
    # ____________________________________________________________________________________________________
    # activation_25 (Activation)       (None, 10, 200, 50)   0           convolution2d_25[0][0]
    # ____________________________________________________________________________________________________
    # convolution2d_26 (Convolution2D) (None, 5, 200, 50)    5005        activation_25[0][0]
    # ____________________________________________________________________________________________________
    # activation_26 (Activation)       (None, 5, 200, 50)    0           convolution2d_26[0][0]
    # ____________________________________________________________________________________________________
    # maxpooling2d_13 (MaxPooling2D)   (None, 5, 100, 25)    0           activation_26[0][0]
    # ____________________________________________________________________________________________________
    # reshape_13 (Reshape)             (None, 12500)         0           maxpooling2d_13[0][0]
    # ____________________________________________________________________________________________________
    # dense_25 (Dense)                 (None, 5)             62505       reshape_13[0][0]
    # ____________________________________________________________________________________________________
    # dense_26 (Dense)                 (None, 1)             6           dense_25[0][0]
    # ====================================================================================================
    # Total params: 69026
    # ____________________________________________________________________________________________________




    # =================================================================================================================
    # Task 4
    # Briefly (not more than half a page) discuss the results you have obtained. Include comparison of each model for
    # their accuracy as well as computational complexity.
    # =================================================================================================================

    # Ranking: CNN, Logistic regression, MLP
    # Computational complexity: Logistic regression, MLP, CNN
    # cnn takes longer for convergence
