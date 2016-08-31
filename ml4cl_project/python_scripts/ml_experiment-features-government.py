
import numpy as np
import os
from collections import Counter
from nltk import ngrams
from sklearn import svm, cross_validation, preprocessing
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import StratifiedKFold
import pickle


# =====================================================================================================================
# Data Extraction
# =====================================================================================================================

def read_all_files_as_gov_or_opp(cur_dir, index=5):
    """
    Read all files of political speeches in a directory recursively and distinguish between government or opposition
    :param cur_dir: directory with political speeches
    :param index: index of government / opposition information in the file name (1 or 0)
    :return: list of file names, government speeches, opposition speeches, speech dictionary
    """

    # get a listing of all plain txt files in the dir and all sub dirs
    txt_file_list = []
    for root, dirs, files in os.walk(cur_dir):
        for name in files:
            if name.endswith('.txt') and not name.startswith('.'):
                txt_file_list.append(os.path.join(root, name))

    # process all files
    type_occurrences = {}
    gov_content = []
    opp_content = []
    for txt_file in txt_file_list:
        # read file content
        cur_content = read_file(txt_file)
        # count type occurrences
        for type in set(cur_content.split()):
            if type in type_occurrences.keys():
                type_occurrences[type] += 1
            else:
                type_occurrences[type] = 1
        # save all speeches to their according list
        isGov = int(txt_file[txt_file.rfind('/')+1:].split('_')[index])
        if isGov:
            gov_content.append(cur_content)
        else:
            opp_content.append(cur_content)

    return txt_file_list, gov_content, opp_content, type_occurrences


def read_file(input_file):
    """
    Retrieves the content of a file as a string object
    :param input_file: file to be read
    :return: file content
    """

    # get file content
    with open(input_file, 'r', encoding="utf-8") as input_stream:
        text = input_stream.read()
    input_stream.close()

    return text


# =====================================================================================================================
# Feature Extraction
# =====================================================================================================================

def get_glove_embeddings(glove_dir="rsrc/glove.6B/glove.6B.50d.txt"):
    """
    Returns glove embeddings as a dictionary of embeddings
    :param glove_dir: directory of the embeddings file
    :return: word embeddings in a dictionary, word vector length
    """

    dict_embeddings = {}
    glove_file = open(glove_dir, 'r')
    # save all embeddings as dictionary with np array values
    for line in glove_file.readlines():
        entry = line.split()
        dict_embeddings[entry[0]] = np.array([float(value) for value in entry[1:]])
    glove_file.close()

    return dict_embeddings, len(entry[1:])


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


def get_matrix_of_averaged_document_embeddings(embeddings, n_dim, texts, stop_words=[''], scale=False):
    """
    Average all word embeddings for a document to a single word embedding per document
    :param embeddings: word embedding dictionary
    :param n_dim: word embedding length
    :param input: Input to be summarized as averaged word embeddings
    :return: Matrix of averaged word embeddings
    """

    matrix = np.zeros((len(texts), n_dim))
    max_abs_scaler = preprocessing.MaxAbsScaler()

    for i_texts in range(0, len(texts)):
        tok = texts[i_texts].split()
        unis = Counter(ngrams(tok, 1))
        n_counted_toks = 0
        for uni in unis:
            if uni[0] in stop_words:
                continue
            if uni[0] in embeddings.keys():
                cur_embedding = max_abs_scaler.fit_transform(embeddings[uni[0]]) if scale else embeddings[uni[0]]
                matrix[i_texts] += cur_embedding * unis[uni]  # sum up
            # else:
            #    continue
        matrix[i_texts] /= len(unis)  # average

        # print("counted: " + str(n_counted_toks))
        # print("printed: " + str(len(unis)))

    return matrix


# =====================================================================================================================
# Linear SVMs
# =====================================================================================================================

def svc(X, y, kernel='linear', c=1.0, probability=False, shrinking=True):
    """
    Perfroms 10 folds cross-validation with the sklearn SVC support vector machine using a linear kernel.
    Performance is evaluated with weighted f1 scores.
    :param X: features
    :param y: classes
    :return: weighted f1 scores
    """

    clf = svm.SVC(kernel=kernel, C=c, probability=probability, shrinking=shrinking)
    return cross_validation.cross_val_score(clf, X, y, scoring='f1_weighted')

# C: Penalty parameter C of the error term. It also controls the trade off between smooth decision boundary and
# classifying the training points correctly.
# c range between 0 and 1000, step size 1


# =====================================================================================================================
# Main
# =====================================================================================================================

if __name__ == '__main__':
    """
    This file contains classification code which is part of the ML4CL term paper
    Task: government vs. opposition (binary) classification
    Features: GloVe and polyglot word embeddings, uni- and bigram counts, type presense, modified with various filters
    ML algorithm: SVM with linear kernel
    """

    # set seed for reproducibility
    seed = 4050712
    np.random.seed(seed)

    # get data
    # dir = '/Volumes/DOCS/Uni/9_SS16/iscl-s9_ss16-machine_learning/project/ml-analysis/toy/'  # toy1
    dir = '/Volumes/DOCS/Uni/9_SS16/iscl-s9_ss16-machine_learning/project/data/bundesparser-plain_text/lp13/' # toy2
    _, gov_speeches, opp_speeches, type_occurrences = read_all_files_as_gov_or_opp(dir)

    speeches = gov_speeches + opp_speeches

    # make labels for prediction
    y = [True] * len(gov_speeches) + [False] * len(opp_speeches)

    # get word vectors for words contained in reviews
    dict_glove_embeddings, n_dim_glove = get_glove_embeddings()
    dict_polyglot_embeddings, n_polyglot_dim = get_polyglot_embeddings()

    # get stop words, i.e. overly common words: skip words occurring in more than every second document
    stop = []
    for key in type_occurrences.keys():
        if type_occurrences[key] > (len(speeches)/1.5):
        # if type_occurrences[key] > (len(speeches)/2):
            stop.append(key)
    len(stop)

    # ================================================================================================================
    # Experiment 1: Use unigram English word vector averaged for documents with default SVM
    # ================================================================================================================

    # 1.1 use simple word vector
    # get features
    sim_glove_pred = get_matrix_of_averaged_document_embeddings(dict_glove_embeddings, n_dim_glove, speeches)
    # create an SVM with a linear kernel
    print(svc(sim_glove_pred, y).mean())

    # 1.2 use word vector without stop words
    # get features
    red_glove_pred = get_matrix_of_averaged_document_embeddings(dict_glove_embeddings, n_dim_glove, speeches, stop)
    # create an SVM with a linear kernel
    print(svc(red_glove_pred, y).mean())


    # ================================================================================================================
    # Experiment 2: Use unigram German word vector averaged for documents with default SVM
    # ================================================================================================================

    # 1.1 use simple word vector
    # get features
    sim_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    # create an SVM with a linear kernel
    print(svc(sim_poly_pred, y).mean())

    # 1.2 use word vector without stop words
    # get features
    red_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches, stop)
    # create an SVM with a linear kernel
    print(svc(red_poly_pred, y).mean())


    # =================================================================================================================
    # Experiment 3: Use both word unigrams, and word bigrams with default SVM
    # =================================================================================================================

    # get ngrams
    ngram_count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    ngram_counts = ngram_count_vectorizer.fit_transform(speeches)
    # create an SVM with a linear kernel
    print(svc(ngram_counts, y).mean())


    # Preprocessing for word embeddings

    # * get rid of overly common words: skip words occurring in more than every second document
    # * keep inflections and letter case, because polyglott is inflected and case sensitive
