__author__ = 'zweiss'

from embeddings import *
from svc import *
from data_extraction import *
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
    seed = 4050712  # set seed for reproducibility
    np.random.seed(seed)

    # get data
    pdir = './rsrc/government-embeddings/'
    # dir = '/Users/zweiss/Desktop/bundesparser-plain_text/data/'
    # _, gov_speeches, opp_speeches, type_occurrences = read_all_files_as_gov_or_opp(dir)
    # pickle.dump((gov_speeches, opp_speeches, type_occurrences), open(pdir+'speeches_binary.p', "wb"))
    gov_speeches, opp_speeches, type_occurrences = pickle.load(open(pdir+'speeches_binary.p', "rb"))
    s_size = len(opp_speeches)
    speeches = random.sample(gov_speeches, s_size) + opp_speeches
    # speeches = gov_speeches + opp_speeches
    print("{} speech(es) acquired.".format(len(speeches)))
    print("{} government speech(es)".format(len(gov_speeches)))
    print("{} opposition speech(es)".format(len(opp_speeches)))

    # make labels for prediction
    y = np.array([1] * s_size + [0] * s_size)
    # y = np.array([1] * len(gov_speeches) + [0] * len(opp_speeches))

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
    len(stop_2)  # 43
    len(stop_15)  # 15

    # get statistics on text length
    tokenizer = WordPunctTokenizer()
    speeches_length = []  # tokenize speeches
    for s in speeches:
        speeches_length.append(len(tokenizer.tokenize(s)))
    speeches_length = np.array(speeches_length)             # all       gov         opp
    speeches_length.min()                                   # 2         2           3
    speeches_length.max()                                   # 13,996    13,996      9,369
    speeches_length.mean()                                  # 496.01    507.34      485.54
    np.median(speeches_length)                              # 223.0     233.0       209.0

    # ================================================================================================================
    # Experiment 1: Compare averaged and concatenated word embeddings
    # ================================================================================================================

    # 1.1 Concatenated word embeddings
    # ================================================================================================================

    # get features and create an SVM with a linear kernel
    # sim_concatenated_pred = get_matrix_of_concatenated_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    #pickle.dump(sim_concatenated_pred, open(pdir+'sim_concatenated_pred.p', "wb"))
    # sim_concatenated_pred = pickle.load(open(pdir+'sim_concatenated_pred.p', "rb"))
    # sim_concatenated_pre, sim_concatenated_rec, sim_concatenated_f1, sim_concatenated_cm = build_and_evaluate_linearSVC(sim_concatenated_pred, y, [1, 0], "binary")
    # pickle.dump((sim_concatenated_pre, sim_concatenated_rec, sim_concatenated_f1, sim_concatenated_cm), open(pdir+'sim_concatenated_results.p', "wb"))
    sim_concatenated_pre, sim_concatenated_rec, sim_concatenated_f1, cm_concatenated = pickle.load(open(pdir+'sim_concatenated_results.p', "rb"))
    cm_concatenated = average_fold_performance(sim_concatenated_cm, 2, 2)

    # report results
    print("Exp. 1.1: Simple SVM with concatenated embeddings")
    print("Avg. F1: {}".format(sim_concatenated_f1.mean()))          # 0.6316580513294853
    print("Avg. Recall: {}".format(sim_concatenated_rec.mean()))     # 0.6553431456927891
    print("Avg. Precision: {}".format(sim_concatenated_pre.mean()))  # 0.6098405278733668
    print("Avg. CM: {}".format(cm_concatenated))
    # [[ 3773.4  1984.5]
    # [ 2417.6  3340.3]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_concatenated, title='Simple linear SVM with concatenated embeddings')
    plt.savefig('rval/government-embeddings-conatenated.png', bbox_inches='tight')

    # 1.2 Averaged word embeddings
    # ================================================================================================================

    # create an SVM with a linear kernel
    # sim_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(sim_poly_pred, open(pdir+'sim_poly_pred.p', "wb"))
    # sim_poly_pred = pickle.load(open(pdir+'sim_poly_pred.p', "rb"))
    # sim_poly_pre, sim_poly_rec, sim_poly_f1, sim_poly_cm = build_and_evaluate_linearSVC(sim_poly_pred, y, [1, 0], "binary")
    # pickle.dump((sim_poly_pre, sim_poly_rec, sim_poly_f1, sim_poly_cm), open(pdir+'sim_poly_results.p', "wb"))
    sim_poly_pre, sim_poly_rec, sim_poly_f1, sim_poly_cm = pickle.load(open(pdir+'sim_poly_results.p', "rb"))

    #  get features and report results
    print("Exp. 1.2: Simple SVM with averaged embeddings")
    print("Avg. F1: {} [+/- {}]".format(sim_poly_f1.mean(), sim_poly_f1.std()))         # 0.6705311132327743
    print("Avg. Recall: {} [+/- {}]".format(sim_poly_rec.mean(), sim_poly_rec.std()))    # 0.6864309622494397
    print("Avg. Precision: {} [+/- {}]".format(sim_poly_pre.mean(), sim_poly_pre.std())) # 0.6556299541330135
    print("Avg. CM: {}".format(average_fold_performance(sim_poly_cm, 2, 2)))
    # [[ 3952.4  1805.5]
    # [ 2081.1  3676.8]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], average_fold_performance(sim_poly_cm, 2, 2), title='Simple linear SVM with averaged embeddings')
    plt.savefig('rval/government-embeddings-averaged.png', bbox_inches='tight')

    # 1.3 Dummy classifier
    # ================================================================================================================

    # create an SVM with a linear kernel
    sim_poly_pred = pickle.load(open(pdir+'sim_poly_pred.p', "rb"))
    dummy_pre, dummy_rec, dummy_f1, dummy_cm = build_and_evaluate_dummy_classifier(sim_poly_pred, y, [1, 0], "binary", strategy='stratified')
    cm_dummy = average_fold_performance(dummy_cm, 2, 2)

    #  get features and report results
    print("Exp. 1.3: Dummy")
    print("Avg. F1: {}".format(dummy_f1.mean()))            # 0.500878673539791
    print("Avg. Recall: {}".format(dummy_rec.mean()))       # 0.5001825525782134
    print("Avg. Precision: {}".format(dummy_pre.mean()))    # 0.501601361897875
    print("Avg. CM: {}".format(cm_dummy))
    # [[ 2880.   2877.9]
    # [ 2861.6  2896.3]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_dummy, title='Dummy classifier')
    plt.savefig('rval/government-dummy.png', bbox_inches='tight')

    # 1.4 Results
    # ================================================================================================================
    print("Results: Concatenation in a dimensionality that is still feasible is significantly worse than averaging.")
    print("Conclusion: Average word embeddings to document level.")

    # compare average with dummy
    print(plot_ttest_results(stats.ttest_rel(sim_poly_f1, dummy_f1)))  # p = 2.935862432567183e-12, df = 49.261364450631646, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(sim_poly_pre, dummy_pre)))  # p = 1.286099123511664e-09, df = 24.94001840469074, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(sim_poly_rec, dummy_rec)))  # p = 6.686606251674593e-15, df = 96.97568244881334, p <= (.05, .01) = (True, True)

    # compare concatenated with dummy
    ttest_dummy_vs_conc = stats.ttest_rel(dummy_f1, sim_concatenated_f1)
    print(plot_ttest_results(ttest_dummy_vs_conc))
    # p = 1.1356298464075644e-11, df = -42.361537934355546, p <= (.05, .01) = (True, True)

    # compare average with concatenated
    ttest_avg_vs_conc = stats.ttest_rel(sim_poly_f1, sim_concatenated_f1)
    print(plot_ttest_results(ttest_avg_vs_conc))
    # p = 6.434174665788181e-06, df = 9.315709751786272, p <= (.05, .01) = (True, True)


    # ================================================================================================================
    # Experiment 2: Compare stemming
    # ================================================================================================================

    # 2.1 German Polyglot embeddings stemmed
    # ================================================================================================================

    # create an SVM with a linear kernel
    # stem_poly_pred = get_matrix_of_averaged_document_embeddings_stemmed(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(stem_poly_pred, open(pdir+'stem_poly_pred.p', "wb"))
    # stem_poly_pred = pickle.load(open(pdir+'stem_poly_pred.p', "rb"))
    # stem_poly_pre, stem_poly_rec, stem_poly_f1, stem_poly_cm = build_and_evaluate_linearSVC(stem_poly_pred, y, [1, 0], "binary")
    # cm_poly_stem = average_fold_performance(stem_poly_cm, 2, 2)
    # pickle.dump((stem_poly_pre, stem_poly_rec, stem_poly_f1, cm_poly_stem), open(pdir+'stem_poly_results.p', "wb"))
    stem_poly_pre, stem_poly_rec, stem_poly_f1, cm_poly_stem = pickle.load(open(pdir+'stem_poly_results.p', "rb"))

    #  get features and report results
    print("Exp. 2.1: Simple SVM with German embeddings stemmed")    # non stemmed embeddings
    print("Avg. F1: {}".format(stem_poly_f1.mean()))                # 0.6698594661418362
    print("Avg. Recall: {}".format(stem_poly_rec.mean()))           # 0.6863094344936588
    print("Avg. Precision: {}".format(stem_poly_pre.mean()))        # 0.6542556586166409
    print("Avg. CM: {}".format(cm_poly_stem))
    # [[ 3951.7  1806.2]
    # [ 2089.7  3668.2]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_poly_stem, title='Simple linear SVM with word embeddings (stemmed)')
    plt.savefig('rval/government-embeddings-stemmed.png', bbox_inches='tight')

    # 2.2 German stemmed Polyglot embeddings stemmed
    # ================================================================================================================

    # create an SVM with a linear kernel
    # stem2_poly_pred = get_matrix_of_averaged_document_embeddings_stemmed(dict_polyglot_stemmed_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(stem2_poly_pred, open(pdir+'stem2_poly_pred.p', "wb"))
    # stem2_poly_pred = pickle.load(open(pdir+'stem2_poly_pred.p', "rb"))
    # stem2_poly_pre, stem2_poly_rec, stem2_poly_f1, stem2_poly_cm = build_and_evaluate_linearSVC(stem2_poly_pred, y, [1, 0], "binary")
    # cm_poly_stem2 = average_fold_performance(stem2_poly_cm, 2, 2)
    # pickle.dump((stem2_poly_pre, stem2_poly_rec, stem2_poly_f1, cm_poly_stem2), open(pdir+'stem2_poly_results.p', "wb"))
    stem2_poly_pre, stem2_poly_rec, stem2_poly_f1, cm_poly_stem2 = pickle.load(open(pdir+'stem2_poly_results.p', "rb"))

    #  get features and report results
    print("Exp. 2.2: Simple SVM with stemmed German embeddings stemmed")    # stemmed embeddings
    print("Avg. F1: {}".format(stem2_poly_f1.mean()))                       # 0.6677098712392711
    print("Avg. Recall: {}".format(stem2_poly_rec.mean()))                  # 0.6845379830573688
    print("Avg. Precision: {}".format(stem2_poly_pre.mean()))               # 0.6519539638047305
    print("Avg. CM: {}".format(cm_poly_stem2))
    # [[ 3941.5  1816.4]
    # [ 2108.6  3649.3]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_poly_stem2, title='Simple linear SVM with stem embeddings (stemmed)')
    plt.savefig('rval/government-embeddings-stemmed2.png', bbox_inches='tight')

    # 2.3 Results
    # ================================================================================================================
    print("Results: Stemming has no positive effect in either configuration compared to non-stemmed embeddings.")
    print("Conclusion: Do not stem (it's a waste of effort).")

    # ================================================================================================================
    # Experiment 3: Use German embeddings with stop words
    # ================================================================================================================

    # 3.1 Stop word criterion == occurs at least once in 50% of all documents
    # ================================================================================================================

    # get features and create an SVM with a linear kernel
    # red2_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches, stop_2)
    # red2_poly_pre, red2_poly_rec, red2_poly_f1, red2_poly_cm = build_and_evaluate_linearSVC(red2_poly_pred, y, [1, 0], "binary")
    # cm_poly2 = average_fold_performance(red2_poly_cm, 2, 2)
    # pickle.dump((red2_poly_pre, red2_poly_rec, red2_poly_f1, cm_poly2), open(pdir+'red2_poly_results.p', "wb"))
    red2_poly_pre, red2_poly_rec, red2_poly_f1, cm_poly2 = pickle.load(open(pdir+'red2_poly_results.p', "rb"))

    # report results
    print("Exp. 3.1: Simple SVM with German embeddings excluding stop words (threshold = 0.5)")
    print("Avg. F1: {}".format(red2_poly_f1.mean()))  # 0.6407528990349265
    print("Avg. Recall: {}".format(red2_poly_rec.mean()))  # 0.6578441829850523
    print("Avg. Precision: {}".format(red2_poly_pre.mean()))  # 0.6250636720522451
    print("Avg. CM:{}".format(cm_poly2))
    # [[ 3787.8  1970.1]
    # [ 2281.   3476.9]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_poly2, title='Simple linear SVM with stop words (t=0.5)')
    plt.savefig('rval/government-embeddings-stemmed2.png', bbox_inches='tight')

    # 3.2 Stop word criterion == occurs at least once in 66.67 percent of all documents
    # ================================================================================================================

    # get features and create an SVM with a linear kernel
    # red15_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches, stop_15)
    # red15_poly_pre, red15_poly_rec, red15_poly_f1, red15_poly_cm = build_and_evaluate_linearSVC(red15_poly_pred, y, [1, 0], "binary")
    # cm_poly15 = average_fold_performance(red15_poly_cm, 2, 2)
    # pickle.dump((red15_poly_pre, red15_poly_rec, red15_poly_f1, cm_poly15), open(pdir+'red15_poly_results.p', "wb"))
    red15_poly_pre, red15_poly_rec, red15_poly_f1, cm_poly15 = pickle.load(open(pdir+'red15_poly_results.p', "rb"))

    # report results
    print("Exp. 3.2: Simple SVM with German embeddings excluding stop words (threshold = 0.67)")
    print("Avg. F1: {}".format(red15_poly_f1.mean()))  # 0.6589959506838013
    print("Avg. Recall: {}".format(red15_poly_rec.mean()))  # 0.6701055959602286
    print("Avg. Precision: {}".format(red15_poly_pre.mean()))  # 0.6486508682056319
    print("Avg. CM: {}".format(cm_poly15))
    # [[ 3858.4  1899.5]
    # [ 2096.8  3661.1]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_poly15, title='Simple linear SVM with stop words (t=0.67)')
    plt.savefig('rval/government-embeddings-stemmed15.png', bbox_inches='tight')

    # 3.3 Results
    # ================================================================================================================
    print("Results: Stop words do not improve the performance. t = .5 decreases it and t = .67 performs equally good.")
    print("Conclusion: Do not use a stop word filter.")


    # ================================================================================================================
    # Experiment 4: Tune the parameters for the support vector machine
    # ================================================================================================================

    # get features
    # sim_poly_pred = get_matrix_of_averaged_document_embeddings(dict_polyglot_embeddings, n_polyglot_dim, speeches)
    # pickle.dump(sim_poly_pred, open(pdir+'sim_poly_pred.p', "wb"))
    sim_poly_pred = pickle.load(open(pdir+'sim_poly_pred.p', "rb"))

    # 4.1 Tune the loss function
    # ================================================================================================================

    # loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
    # Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’
    # is the square of the hinge loss.

    spoly_hinge_pre, spoly_hinge_rec, spoly_hinge_f1, spoly_hinge_cm = build_and_evaluate_linearSVC(
        sim_poly_pred, y, [1, 0], "binary", num_folds=10, penalty='l2', loss='hinge', dual=True, tol=0.0001, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None)

    # report results
    print("Exp. 3.1: Use hinge loss function instead of squared hinge")
    print("Avg. F1: {}".format(spoly_hinge_f1.mean()))  # 0.6710829851559729
    print("Avg. Recall: {}".format(spoly_hinge_rec.mean()))  # 0.7066083380200112
    print("Avg. Precision: {}".format(spoly_hinge_pre.mean()))  # 0.6405338692541134
    cm_spoly_hinge = np.zeros((2,2))
    for mtrx in spoly_hinge_cm:
        cm_spoly_hinge += mtrx
    cm_spoly_hinge = cm_spoly_hinge/10
    print("Avg. CM: {}".format(cm_spoly_hinge))
    # [[ 3270.6  2488.8]
    # [ 1838.9  4428.8]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_spoly_hinge)

    # Conclusion:
    # slightly better, but probably not significant. Note: although, squared hinge was used as default here, usually
    # hinge is the default. So I'll report that the squared hinge did not improve performance

    # 4.2 Tune optimization problem
    # ================================================================================================================

    # dual : bool, (default=True)
    # Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when
    # n_samples > n_features.
    print("n_samples ({}) > n_features ({}) suggests dual = False".format(sim_poly_pred.shape[0], sim_poly_pred.shape[1]))

    spoly_nondual_pre, spoly_nondual_rec, spoly_nondual_f1, spoly_nondual_cm = build_and_evaluate_linearSVC(
        sim_poly_pred, y, [1, 0], "binary", num_folds=10, penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None)
    # note: have to use squared_hinge here: The combination of penalty='l2' and loss='hinge' are not supported when
    # dual=False, Parameters: penalty='l2', loss='hinge', dual=False

    # report results
    print("Exp. 3.2: Use non dual optimization problem")
    print("Avg. F1: {}".format(spoly_nondual_f1.mean()))  # 0.6713783245227163
    print("Avg. Recall: {}".format(spoly_nondual_rec.mean()))  # 0.7085867347006316
    print("Avg. Precision: {}".format(spoly_nondual_pre.mean()))  # 0.639466876610021
    cm_spoly_nondual = np.zeros((2,2))
    for mtrx in spoly_nondual_cm:
        cm_spoly_nondual += mtrx
    cm_spoly_nondual = cm_spoly_nondual/10
    print("Avg. CM: {}".format(cm_spoly_nondual))
    # [[ 3251.7  2507.7]
    # [ 1826.5  4441.2]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_spoly_nondual)

    # Conclusion:
    # no real difference, but since the documentation suggests to not use a dual optimization problem, when the number
    # of samples exceeds the number of features, which is the case here, I will stay with dual=False. As this requires
    # a squared_hinge and there was no real difference between the loss functions, this means we will use hinge^2 from
    # now on.

    # 4.3 Tune mutli class stratedy
    # ================================================================================================================

    # multi_class: string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’) :
    # Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest
    # classifiers, while "crammer_singer" optimizes a joint objective over all classes. While crammer_singer is
    # interesting from a theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads
    # to better accuracy and is more expensive to compute. If "crammer_singer" is chosen, the options loss, penalty and
    # dual will be ignored.

    spoly_crams_pre, spoly_crams_rec, spoly_crams_f1, spoly_crams_cm = build_and_evaluate_linearSVC(
        sim_poly_pred, y, [1, 0], "binary", num_folds=10, penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None, multi_class="crammer_singer")

    # report results
    print("Exp. 3.3: Use crammer singer multi class stratedy")
    print("Avg. F1: {}".format(spoly_crams_f1.mean()))  # 0.6708540138541019
    print("Avg. Recall: {}".format(spoly_crams_rec.mean()))  # 0.7057627274235265
    print("Avg. Precision: {}".format(spoly_crams_pre.mean()))  # 0.6408209952944942
    cm_spoly_crams = np.zeros((2,2))
    for mtrx in spoly_crams_cm:
        cm_spoly_crams += mtrx
    cm_spoly_crams = cm_spoly_crams/10
    print("Avg. CM: {}".format(cm_spoly_crams))
    # [[ 3276.6  2482.8]
    # [ 1844.2  4423.5]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_spoly_crams)

    # Conclusion:
    # no real difference. As Crammer Singer is rather unsual to use and makes loss, penalty and dual obsolete, it will
    # not be used

    # 4.4 Tune penalty function
    # ================================================================================================================

    # penalty : string, ‘l1’ or ‘l2’ (default=’l2’)
    # Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to
    # coef_ vectors that are sparse.

    spoly_l1_pre, spoly_l1_rec, spoly_l1_f1, spoly_l1_cm = build_and_evaluate_linearSVC(
        sim_poly_pred, y, [1, 0], "binary", num_folds=10, penalty='l1', loss='squared_hinge', dual=False, tol=0.0001, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None)

    # report results
    print("Exp. 3.4: Use l1 regularisation instead of l2 regularisation")
    print("Avg. F1: {}".format(spoly_l1_f1.mean()))  # 0.6714532033038876
    print("Avg. Recall: {}".format(spoly_l1_rec.mean()))  # 0.708554816413077
    print("Avg. Precision: {}".format(spoly_l1_pre.mean()))  # 0.6396424778207841
    cm_spoly_l1= np.zeros((2,2))
    for mtrx in spoly_l1_cm:
        cm_spoly_l1 += mtrx
    cm_spoly_l1 = cm_spoly_l1/10
    print("Avg. CM: {}".format(cm_spoly_l1))
    # [[ 3253.5  2505.9]
    # [ 1826.7  4441. ]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_spoly_l1)

    # Conclusion:
    # no relevant difference, so stay with the default L2. Is also more efficient with non-sparse data, as I have here.

    # 4.5 Penalty parameter
    # ================================================================================================================

    # C : float, optional (default=1.0)
    # Penalty parameter C of the error term.

    spoly_pre_c15, spoly_rec_c15, spoly_f1_c15, spoly_cm_c15 = build_and_evaluate_linearSVC(
        sim_poly_pred, y, [1, 0], "binary", num_folds=10, penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=.1,
        fit_intercept=True, intercept_scaling=1, class_weight=None)
    print(spoly_f1_c15.mean())

    # C=1^-5    0.685189234066
    # C=.1      0.671349801098
    # C=1       0.67139400839
    # C=1.5     0.671367246008
    # C=10      0.671479482463
    # C=100     0.671481799681

    # report results
    print("Exp. 3.5: Use penalty parameter C=.000001")
    print("Avg. F1: {}".format(spoly_f1_c15.mean()))  # 0.6714532033038876
    print("Avg. Recall: {}".format(spoly_l1_rec.mean()))  # 0.708554816413077
    print("Avg. Precision: {}".format(spoly_pre_c15.mean()))  # 0.6396424778207841
    cm_spoly= np.zeros((2,2))
    for mtrx in spoly_cm_c15:
        cm_spoly += mtrx
    cm_spoly = cm_spoly/10
    print("Avg. CM: {}".format(cm_spoly))
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_spoly)

    # Conclusion:
    # very low C values decrease precision, but also increase recall (as to be expected). However, the f1 Measure stays
    # constant, so we keep C=1

    # 4.6 Tolerance for stopping criteria
    # ================================================================================================================

    # tol : float, optional (default=1e-4)
    # Tolerance for stopping criteria.

    spoly_pre_tol, spoly_rec_tol, spoly_f1_tol, spoly_cm_tol = build_and_evaluate_linearSVC(
        sim_poly_pred, y, [1, 0], "binary", num_folds=10, penalty='l2', loss='squared_hinge', dual=False, tol=.0005, C=1,
        fit_intercept=True, intercept_scaling=1, class_weight=None)
    print(spoly_f1_tol.mean())

    # tol=1e-200            0.671409608689
    # tol=1e-100            0.671409608689
    # tol=0.0001 (default)  0.671388250178
    # tol=0.0005            0.671430276981
    # tol=0.001             0.671435188802
    # tol=0.01              0.671346084388
    # tol=0.1               0.645250022795
    # tol=100               0

    # decreasing the tolerance increases f1 score, but only slightly, increasing it worse. Just stay with the default



    # Preprocessing for word embeddings

    # * get rid of overly common words: skip words occurring in more than every second document
    # * keep inflections and letter case, because polyglott is inflected and case sensitive
