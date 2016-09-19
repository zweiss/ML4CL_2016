__author__ = 'zweiss'

import numpy as np
from collections import Counter
from nltk import ngrams
from nltk.stem import *
from nltk.tokenize import WordPunctTokenizer
from sklearn import preprocessing
import pickle


# =====================================================================================================================
# Get Embeddings
# =====================================================================================================================

# def get_glove_embeddings(glove_dir="rsrc/glove.6B/glove.6B.50d.txt"):
#     """
#     Returns glove embeddings as a dictionary of embeddings
#     :param glove_dir: directory of the embeddings file
#     :return: word embeddings in a dictionary, word vector length
#     """
#
#     dict_embeddings = {}
#     glove_file = open(glove_dir, 'r')
#     # save all embeddings as dictionary with np array values
#     for line in glove_file.readlines():
#         entry = line.split()
#         dict_embeddings[entry[0]] = np.array([float(value) for value in entry[1:]])
#     glove_file.close()
#
#     return dict_embeddings, len(entry[1:])


def get_polyglot_embeddings(polyglot_dir="rsrc/polyglot-de.pkl"):
    """
    Get German polyglot word embeddings in dictionary format (https://sites.google.com/site/rmyeid/projects/polyglot/)
    :param polyglot_dir: directory containing the polyglot pickle file
    :return: polyglot word embeddings in dictionary format, embedding length
    """

    words, embeddings = pickle.load(open(polyglot_dir, 'rb'), encoding='latin1')
    # print("Emebddings shape is {}".format(embeddings.shape))
    dict_embeddings = {}
    for i in range(0, len(words)):
        dict_embeddings[words[i]] = embeddings[i]
    # print(len(dict_embeddings.keys()))

    return dict_embeddings, len(embeddings[i])


def get_stemmed_polyglot_embeddings(polyglot_dir="rsrc/polyglot-de.pkl"):
    """
    Get German polyglot word embeddings in dictionary format (https://sites.google.com/site/rmyeid/projects/polyglot/)
    :param polyglot_dir: directory containing the polyglot pickle file
    :return: polyglot word embeddings in dictionary format, embedding length
    """

    words, embeddings = pickle.load(open(polyglot_dir, 'rb'), encoding='latin1')
    stemmer = SnowballStemmer("german")  # stem word embeddings
    dict_embeddings = {}
    embedding_counter = {}
    for i in range(0, len(words)):
        stemmed = stemmer.stem(words[i])
        if stemmed in dict_embeddings.keys():
            dict_embeddings[stemmed] += embeddings[i]
            embedding_counter[stemmed] += 1
        else:
            dict_embeddings[stemmed] = embeddings[i]
            embedding_counter[stemmed] = 1

    # normalize embeddings
    for key in dict_embeddings.keys():
        dict_embeddings[key] /= embedding_counter[key]

    return dict_embeddings, len(embeddings[i])


# =====================================================================================================================
# Get embedding matrices
# =====================================================================================================================

def get_matrix_of_concatenated_document_embeddings(embeddings, n_dim, texts, token_limit=20, stop_words=[''], scale=False):
    """

    :param embeddings:
    :param n_dim:
    :param texts:
    :param n_tokens:
    :param stop_words:
    :param scale:
    :return:
    """

    scaler = preprocessing.MaxAbsScaler()
    # scaler = preprocessing.MinMaxScaler()
    tokenizer = WordPunctTokenizer()

    matrix = np.zeros((len(texts), token_limit*n_dim))
    for i_texts in range(0, len(texts)):
        tokens = tokenizer.tokenize(texts[i_texts])
        tmp = []
        for i_token in range(0, token_limit):
            cur_embedding = [0] * n_dim
            # if text still has tokens left, the current token is in the embeddings, and it is not on the stop word list
            if i_token < len(tokens) and tokens[i_token] in embeddings.keys() and not tokens[i_token] in stop_words:
                tmp_embedding = scaler.fit_transform(embeddings[tokens[i_token]]) if scale else embeddings[tokens[i_token]]
                cur_embedding = tmp_embedding.tolist()
            tmp += cur_embedding

        matrix[i_texts] = np.array(tmp)

    return matrix


def get_matrix_of_averaged_document_embeddings(embeddings, n_dim, texts, stop_words=[''], scale=False, scaler=preprocessing.MaxAbsScaler()):
    """
    Average all word embeddings for a document to a single word embedding per document
    :param embeddings: word embedding dictionary
    :param n_dim: word embedding length
    :param input: Input to be summarized as averaged word embeddings
    :return: Matrix of averaged word embeddings
    """

    matrix = np.zeros((len(texts), n_dim))
    tokenizer = WordPunctTokenizer()

    for i_texts in range(0, len(texts)):
        tok = tokenizer.tokenize(texts[i_texts])
        unis = Counter(ngrams(tok, 1))
        n_counted_toks = 0
        for uni in unis:
            if uni[0] in stop_words:
                continue
            if uni[0] in embeddings.keys():
                cur_embedding = scaler.fit_transform(embeddings[uni[0]]) if scale else embeddings[uni[0]]
                matrix[i_texts] += cur_embedding * unis[uni]  # sum up
            # else:
            #    continue
        matrix[i_texts] /= len(unis)  # average

        # print("counted: " + str(n_counted_toks))
        # print("printed: " + str(len(unis)))

    return matrix


def get_matrix_of_averaged_document_embeddings_stemmed(embeddings, n_dim, texts, stop_words=[''], scale=False, scaler=preprocessing.MaxAbsScaler()):
    """
    Average all word embeddings for a document to a single word embedding per document
    :param embeddings: word embedding dictionary
    :param n_dim: word embedding length
    :param input: Input to be summarized as averaged word embeddings
    :return: Matrix of averaged word embeddings
    """

    matrix = np.zeros((len(texts), n_dim))

    stemmer = SnowballStemmer("german")
    tokenizer = WordPunctTokenizer()


    for i_texts in range(0, len(texts)):

        # preprocessing
        tok = [stemmer.stem(t) for t in tokenizer.tokenize(texts[i_texts])]
        unis = Counter(ngrams(tok, 1))
        n_counted_toks = 0
        for uni in unis:
            tok = stemmer.stem(uni[0])
            if uni[0] in stop_words:
                continue
            if uni[0] in embeddings.keys():
                cur_embedding = scaler.fit_transform(embeddings[uni[0]]) if scale else embeddings[uni[0]]
                matrix[i_texts] += cur_embedding * unis[uni]  # sum up
            # else:
            #    continue
        matrix[i_texts] /= len(unis)  # average

        # print("counted: " + str(n_counted_toks))
        # print("printed: " + str(len(unis)))

    return matrix

