__author__ = 'zweiss'

from data_extraction import *
from embeddings import *
from svc import *
from sklearn.preprocessing import LabelEncoder
import random

import numpy as np
import os
import pickle
from scipy import stats


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

    # set working directory ### CHANGE ###
    os.chdir('/Volumes/INTENSO/ml-project/ml-analysis')
    # set seed for reproducibility
    seed = 4050712
    np.random.seed(seed)

    # get data
    pdir = './rsrc/party-embeddings/'
    # dir = '/Volumes/DOCS/Uni/9_SS16/iscl-s9_ss16-machine_learning/project/data/'
    # _, party_speeches, type_occurrences = read_all_files_as_party(dir)
    # pickle.dump((party_speeches, type_occurrences), open(pdir+'speeches_party.p', "wb"))
    party_speeches, type_occurrences = pickle.load(open(pdir+'speeches_party.p', "rb"))

    speeches = []
    y_cat = []
    keys = [k for k in party_speeches.keys()]
    s_size = 11060
    for k in keys:
        if not k == "none":
            n_speeches = len(party_speeches[k])
            print("{} {} speech(es)".format(n_speeches, k))
            speeches += [s for s in random.sample(party_speeches[k], s_size)]
            y_cat += [k] * s_size
    print("{} speech(es) acquired.".format(len(speeches)))
    len(y_cat)

    lb = LabelEncoder()  # change label encoding
    y = lb.fit_transform(y_cat)

    # get word vectors for words contained in reviews
    dict_polyglot_embeddings, n_polyglot_dim = get_polyglot_embeddings()
    dict_polyglot_stemmed_embeddings, _ = get_stemmed_polyglot_embeddings()

    # get stop words, i.e. overly common words: skip words occurring in more than every second document
    stop_2 = []
    stop_15 = []
    for key in type_occurrences.keys():
        if type_occurrences[key] > (len(speeches)/1.5):
            stop_15.append(key)
        if type_occurrences[key] > (len(speeches)/2):
            stop_2.append(key)
    len(stop_2)  # 141
    len(stop_15)  # 98

    # ================================================================================================================
    # Experiment 1: Compare averaged and concatenated word embeddings
    # ================================================================================================================

    # 1.1 Concatenated word embeddings
    # ================================================================================================================

    # get features and create an SVM with a linear kernel
    # sim_concatenated_pred = get_matrix_of_concatenated_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(sim_concatenated_pred, open(pdir+'sim_concatenated_pred.p', "wb"))
    sim_concatenated_pred = pickle.load(open(pdir+'sim_concatenated_pred.p', "rb"))
    sim_concatenated_pre, sim_concatenated_rec, sim_concatenated_f1, sim_concatenated_cm = build_and_evaluate_linearSVC(sim_concatenated_pred, y, [0, 1, 2, 3, 4], "weighted")
    pickle.dump((sim_concatenated_pre, sim_concatenated_rec, sim_concatenated_f1, sim_concatenated_cm), open(pdir+'sim_concatenated_results.p', "wb"))
    sim_concatenated_pre, sim_concatenated_rec, sim_concatenated_f1, cm_concatenated = pickle.load(open(pdir+'sim_concatenated_results.p', "rb"))
    cm_concatenated = average_fold_performance(cm_concatenated, 5, 5)

    # report results
    print("Exp. 1.1: Simple SVM with concatenated embeddings")
    print("Avg. F1: {}".format(sim_concatenated_f1.mean()))          # 0.2788053426362077
    print("Avg. Recall: {}".format(sim_concatenated_rec.mean()))     # 0.28106690777576854
    print("Avg. Precision: {}".format(sim_concatenated_pre.mean()))  # 0.27937085504943815
    print("Avg. CM: {}".format(cm_concatenated))
    # [[ 274.6  171.4  201.1  215.5  243.4]
    # [ 186.   243.   200.9  247.3  228.8]
    # [ 195.2  153.5  304.2  283.   170.1]
    # [ 139.8  155.2  237.1  418.   155.9]
    # [ 213.7  187.3  183.7  206.8  314.5]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['cdu', 'fdp', 'gruene', 'linke', 'spd'], cm_concatenated, title='Simple linear SVM with concatenated embeddings')
    plt.savefig('rval/party-embeddings-conatenated.png', bbox_inches='tight')

    # 1.2 Averaged word embeddings
    # ================================================================================================================

    # create an SVM with a linear kernel
    sim_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    pickle.dump(sim_poly_pred, open(pdir+'sim_poly_pred.p', "wb"))
    sim_poly_pred = pickle.load(open(pdir+'sim_poly_pred.p', "rb"))
    sim_poly_pre, sim_poly_rec, sim_poly_f1, sim_poly_cm = build_and_evaluate_linearSVC(sim_poly_pred, y, [0, 1, 2, 3, 4], "weighted")
    pickle.dump((sim_poly_pre, sim_poly_rec, sim_poly_f1, sim_poly_cm), open(pdir+'party_sim_poly_results.p', "wb"))
    sim_poly_pre, sim_poly_rec, sim_poly_f1, cm_poly = pickle.load(open(pdir+'party_sim_poly_results.p', "rb"))
    cm_poly = average_fold_performance(sim_poly_cm, 5, 5)

    #  get features and report results
    print("Exp. 1.2: Simple SVM with German embeddings")
    print("Avg. F1: {}".format(sim_poly_f1.mean()))  # 0.2933017213525976
    print("Avg. Precision: {}".format(sim_poly_pre.mean()))  # 0.2996565885369916
    print("Avg. Recall: {}".format(sim_poly_rec.mean()))  # 0.30587703435804703
    print("Avg. CM: {}".format(cm_poly))

    # 1.3 Dummy classifier
    # ================================================================================================================

    # create an SVM with a linear kernel
    sim_poly_pred = pickle.load(open(pdir+'sim_poly_pred.p', "rb"))
    dummy_pre, dummy_rec, dummy_f1, dummy_cm = build_and_evaluate_dummy_classifier(sim_poly_pred, y, [0, 1, 2, 3, 4], "weighted", strategy='stratified')
    cm_dummy = average_fold_performance(dummy_cm, 5, 5)

    #  get features and report results
    print("Exp. 1.3: Dummy")
    print("Avg. F1: {}".format(dummy_f1.mean()))            # 0.19719150571366956
    print("Avg. Recall: {}".format(dummy_rec.mean()))       # 0.1972151898734177
    print("Avg. Precision: {}".format(dummy_pre.mean()))    # 0.19723553612336495
    print("Avg. CM: {}".format(cm_dummy))

    # compare averaged with dummy
    print(plot_ttest_results(stats.ttest_rel(sim_poly_f1, dummy_f1)))  # p = 7.79423786704069e-11, df = 34.15811144266951, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(sim_poly_rec, dummy_pre)))  # p = 2.6130165545096517e-11, df = 38.59744786956253, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(sim_poly_pre, dummy_rec)))  # p = 6.936951490830549e-11, df = 34.60632842594961, p <= (.05, .01) = (True, True)




    # ================================================================================================================
    # Experiment 2: Compare stemming
    # ================================================================================================================

    # 2.1 German Polyglot embeddings stemmed
    # ================================================================================================================

    # create an SVM with a linear kernel
    # stem_poly_pred = get_matrix_of_averaged_document_embeddings_stemmed(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(stem_poly_pred, open(pdir+'stem_poly_pred.p', "wb"))
    # stem_poly_pred = pickle.load(open(pdir+'stem_poly_pred.p', "rb"))
    # stem_poly_pre, stem_poly_rec, stem_poly_f1, stem_poly_cm = build_and_evaluate_linearSVC(stem_poly_pred, y, [0, 1, 2, 3, 4], "weighted")
    # stem_poly_cm = average_fold_performance(stem_poly_cm, 5, 5)
    # pickle.dump((stem_poly_pre, stem_poly_rec, stem_poly_f1, stem_poly_cm), open(pdir+'party_stem_poly_results.p', "wb"))
    stem_poly_pre, stem_poly_rec, stem_poly_f1, stem_poly_cm = pickle.load(open(pdir+'party_stem_poly_results.p', "rb"))

    #  get features and report results
    print("Exp. 2.3: Simple SVM with German embeddings stemmed")
    print("Avg. F1: {}".format(stem_poly_f1.mean()))  # 0.27959199191281037
    print("Avg. Recall: {}".format(stem_poly_rec.mean()))  # 0.29493670886075957
    print("Avg. Precision: {}".format(stem_poly_pre.mean()))  # 0.2866774673880224
    print("Avg. CM: {}".format(stem_poly_cm))

    # 2.2 German stemmed polyglot embeddings stemmed
    # ================================================================================================================

    # create an SVM with a linear kernel
    # stem2_poly_pred = get_matrix_of_averaged_document_embeddings_stemmed(dict_polyglot_stemmed_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(stem2_poly_pred, open(pdir+'stem2_poly_pred.p', "wb"))
    # stem2_poly_pred = pickle.load(open(pdir+'stem2_poly_pred.p', "rb"))
    # stem2_poly_pre, stem2_poly_rec, stem2_poly_f1, stem2_poly_cm = build_and_evaluate_linearSVC(stem2_poly_pred, y, [0, 1, 2, 3, 4], "weighted")
    # cm_poly_stem2 = average_fold_performance(stem2_poly_cm, 5, 5)
    # pickle.dump((stem2_poly_pre, stem2_poly_rec, stem2_poly_f1, cm_poly_stem2), open(pdir+'stem2_poly_results.p', "wb"))
    stem2_poly_pre, stem2_poly_rec, stem2_poly_f1, cm_poly_stem2 = pickle.load(open(pdir+'stem2_poly_results.p', "rb"))

    #  get features and report results
    print("Exp. 2.2: Simple SVM with stemmed German embeddings stemmed")    # stemmed embeddings
    print("Avg. F1: {}".format(stem2_poly_f1.mean()))                       # 0.2918832405671118
    print("Avg. Recall: {}".format(stem2_poly_rec.mean()))                  # 0.3023146473779385
    print("Avg. Precision: {}".format(stem2_poly_pre.mean()))               # 0.29647786458055087
    print("Avg. CM: {}".format(cm_poly_stem2))

    # ================================================================================================================
    # Experiment 3: Use German embeddings with stop words
    # ================================================================================================================

    # 3.1 Stop word criterion == occurs at least once in half of all documents
    # ================================================================================================================

    # get features and create an SVM with a linear kernel
    # red2_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches, stop_2)
    # pickle.dump(red2_poly_pred, open(pdir+'red2_poly_pred.p', "wb"))
    # red2_poly_pred = pickle.load(open(pdir+'red2_poly_pred.p', "rb"))
    # red2_poly_pre, red2_poly_rec, red2_poly_f1, red2_poly_cm = build_and_evaluate_linearSVC(red2_poly_pred, y, [0, 1, 2, 3, 4], "weighted")
    # pickle.dump((red2_poly_pre, red2_poly_rec, red2_poly_f1, red2_poly_cm), open(pdir+'party_red2_poly_results.p', "wb"))
    red2_poly_pre, red2_poly_rec, red2_poly_f1, red2_poly_cm = pickle.load(open(pdir+'party_red2_poly_results.p', "rb"))

    # report results
    print("Exp. 2.1: Simple SVM with German embeddings excluding stop words (threshold = 0.5)")
    print("Avg. F1: {}".format(red2_poly_f1.mean()))  # 0.27983143130105936
    print("Avg. Recall: {}".format(red2_poly_rec.mean()))  # 0.28954792043399635
    print("Avg. Precision: {}".format(red2_poly_pre.mean()))  # 0.2839773168189577

    # 3.2 Stop word criterion == occurs at least once in 66.67 percent of all documents
    # ================================================================================================================

    # get features and create an SVM with a linear kernel
    # red15_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches, stop_15)
    # pickle.dump(red15_poly_pred, open(pdir+'red15_poly_pred.p', "wb"))
    # red15_poly_pred = pickle.load(open(pdir+'red15_poly_pred.p', "rb"))
    # red15_poly_pre, red15_poly_rec, red15_poly_f1, red15_poly_cm = build_and_evaluate_linearSVC(red15_poly_pred, y, [0, 1, 2, 3, 4], "weighted")
    # pickle.dump((red15_poly_pre, red15_poly_rec, red15_poly_f1, red15_poly_cm), open(pdir+'party_red15_poly_results.p', "wb"))
    red15_poly_pre, red15_poly_rec, red15_poly_f1, red15_poly_cm = pickle.load(open(pdir+'party_red15_poly_results.p', "rb"))

    # report results
    print("Exp. 2.2: Simple SVM with German embeddings excluding stop words (threshold = 0.67)")
    print("Avg. F1: {}".format(red15_poly_f1.mean()))  # 0.2822738576029507
    print("Avg. Precision: {}".format(red15_poly_pre.mean()))  # 0.2862946432081048
    print("Avg. Recall: {}".format(red15_poly_rec.mean()))  # 0.2919891500904159


    # 2.3 Results
    # ================================================================================================================
    print("Results: Stop words do not improve the performance. t = .5 decreases it and t = .67 performs equally good.")
    print("Conclusion: Do not use a stop word filter.")

