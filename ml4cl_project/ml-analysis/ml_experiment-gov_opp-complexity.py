__author__ = 'zweiss'

import os
import pandas as pd
from svc import *
from data_extraction import *
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, RFECV, VarianceThreshold
import pickle
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from scipy import stats

# =====================================================================================================================
# Main
# =====================================================================================================================


if __name__ == '__main__':
    """
    This file contains classification code which is part of the ML4CL term paper
    Task: government vs. opposition (binary) classification
    Features: complexity features: hancke, weiss, galasso, min
    ML algorithm: SVM with linear kernel
    """

    # set working directory ### CHANGE ###
    os.chdir('/Volumes/INTENSO/ml-project/ml-analysis/')
    seed = 4050712
    np.random.seed(seed)

    # get data
    pdir = './rsrc/government-complexity/'
    lc_table_raw = pd.read_csv("./rsrc/bundesparser-160903_2_meta-new.csv", low_memory=False)
    lc_table = lc_table_raw[lc_table_raw.isGovernment == 0.0]
    sample_size = lc_table.shape[0]
    lc_table = lc_table.append(lc_table_raw[lc_table_raw.isGovernment == 1.0].sample(sample_size))
    print("{} speech(es) acquired.".format(lc_table.shape[0]))
    print("{} government speech(es)".format(lc_table[lc_table.isGovernment == 1.0].shape[0]))
    print("{} opposition speech(es)".format(lc_table[lc_table.isGovernment == 0.0].shape[0]))

    # make labels for prediction
    y = np.array([1.0] * sample_size + [0.0] * sample_size)

    # get complexity feature sets
    comp_start = 9
    lc_table_no_meta = lc_table.iloc[0:lc_table.shape[0], comp_start:].drop('SENT_SYN_complexTunitRatio', axis=1).drop('SENT_SYN_TunitComplexityRatio', axis=1).drop('COHE_', axis=1)
    pickle.dump((lc_table_no_meta, y), open(pdir+'lc_table_no_meta.p', 'wb'))
    lc_table_no_meta, y = pickle.load(open(pdir+'lc_table_no_meta.p', 'rb'))
    full_re = 'SENT.*|PHRA.*|LEX.*|MORPH.*|COHE.*|PID|Deagent.*'

    # get scaled features in range -1, 1
    lc_matrix_sent_scaled = get_feature_set(lc_table_no_meta, "SENT.*", True)
    lc_matrix_phra_scaled = get_feature_set(lc_table_no_meta, "PHRA.*", True)
    lc_matrix_lex_scaled = get_feature_set(lc_table_no_meta, "LEX.*", True)
    lc_matrix_morph_scaled = get_feature_set(lc_table_no_meta, "MORPH.*", True)
    lc_matrix_deag_scaled = get_feature_set(lc_table_no_meta, "Deag.*", True)
    lc_matrix_cohe_scaled = get_feature_set(lc_table_no_meta, "COHE.*", True)
    lc_matrix_lmor_scaled = get_feature_set(lc_table_no_meta, "LEX_.*|MORPH_.*", True)
    lc_matrix_lmorph_scaled = get_feature_set(lc_table_no_meta, "LEX_.*|MORPH_.*|PHRA.*", True)
    lc_matrix_lmorphcoh_scaled = get_feature_set(lc_table_no_meta, "LEX_.*|MORPH_.*|PHRA.*|COH.*", True)
    lc_matrix_lmorphcohse_scaled = get_feature_set(lc_table_no_meta, "LEX_.*|MORPH_.*|PHRA.*|COH.*|SENT.*", True)
    lc_matrix_full_scaled = get_feature_set(lc_table_no_meta, full_re, True)

    lc_matrix_full = get_feature_set(lc_table_no_meta, full_re, False)
    lc_matrix_full_scaled0 = get_feature_set(lc_table_no_meta, full_re, True, preprocessing.MinMaxScaler())

    # ================================================================================================================
    # Experiment 0: get base lines
    # ================================================================================================================


    # 0. Dummy classifier
    # ================================================================================================================

    # create an SVM with a linear kernel
    dummy_pre, dummy_rec, dummy_f1, dummy_cm = build_and_evaluate_dummy_classifier(np.zeros((len(y),3)), y, [1, 0], "binary", strategy='stratified')
    cm_dummy = average_fold_performance(dummy_cm, 2, 2)

    #  get features and report results
    print("Exp. 1.3: Dummy")
    print("Avg. F1: {}".format(dummy_f1.mean()))            # 0.4986208459147184
    print("Avg. Recall: {}".format(dummy_rec.mean()))       # 0.49870626109429106
    print("Avg. Precision: {}".format(dummy_pre.mean()))    # 0.4985481948226864
    print("Avg. CM: {}".format(cm_dummy))
    # [[ 2880.   2877.9]
    # [ 2861.6  2896.3]]
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(['Government', 'Opposition'], cm_dummy, title='Dummy classifier')
    plt.savefig('rval/government-dummy.png', bbox_inches='tight')

    sim_poly_pre, sim_poly_rec, sim_poly_f1, sim_poly_cm = pickle.load(open('./rsrc/government-embeddings/sim_poly_results.p', "rb"))

    # ================================================================================================================
    # Experiment 1: compare full feature set scaled and unscaled
    # ================================================================================================================

    # 1.1 Full feature set not scaled
    # ================================================================================================================

    # create linear kernel SVM
    # prec_full, rec_full, f1_full, cm_full = build_and_evaluate_linearSVC(lc_matrix_full, y, [1, 0], "binary")
    # pickle.dump((prec_full, rec_full, f1_full, cm_full), open(pdir+'sim_full_results.p', "wb"))
    # cm_full_total = average_fold_performance(cm_full, 2, 2)
    prec_full, rec_full, f1_full, cm_full_total = pickle.load(open(pdir+'sim_full_results.p', "rb"))

    # report results
    print("Exp. 1.1: Simple SVM with full feature set (not scaled)")
    print("Avg. F1: {} [+/- {}]".format(f1_full.mean(), f1_full.std()))  # Avg. F1: 0.611351622719487 [+/- 0.01866259210997136]
    print("Avg. Recall: {} [+/- {}]".format(rec_full.mean(), rec_full.std()))  # Avg. Recall: 0.598637769939587 [+/- 0.028963909175201554]
    print("Avg. Precision: {} [+/- {}]".format(prec_full.mean(), prec_full.std()))  # Avg. Precision: 0.6250718780116866 [+/- 0.00884554195126555]
    print("Avg. CM:{}".format(cm_full_total))
    # [[ 3446.9  2311. ]
    # [ 2065.6  3692.3]]

    # 1.2 Full feature set scaled from -1 to 1
    # ================================================================================================================

    # create linear kernel SVM
    # prec_full_scaled, rec_full_scaled, f1_full_scaled, cm_full_scaled = build_and_evaluate_linearSVC(lc_matrix_full_scaled, y, [1, 0], "binary")
    # pickle.dump((prec_full_scaled, rec_full_scaled, f1_full_scaled, cm_full_scaled), open(pdir+'scaled_full_results.p', "wb"))
    prec_full_scaled, rec_full_scaled, f1_full_scaled, cm_full_scaled_total = pickle.load(open(pdir+'scaled_full_results.p', "rb"))
    cm_full_scaled_total = average_fold_performance(cm_full_scaled_total, 2, 2)

    # report results
    print("Exp. 1.2: Simple SVM with full feature set (scaled -1 to 1)")
    print("Avg. F1: {} [+/- {}]".format(f1_full_scaled.mean(), f1_full_scaled.std()))  # 0.6154393457306255 [+/- 0.019714729936151452]
    print("Avg. Recall: {} [+/- {}]".format(rec_full_scaled.mean(), rec_full_scaled.std()))  # 0.5994714289256754 [+/- 0.029627079555298946]
    print("Avg. Precision: {} [+/- {}]".format(prec_full_scaled.mean(), prec_full_scaled.std()))  # 0.6327167740202789 [+/- 0.01057203149063686]


    # run ml_experiment-gov_opp-embeddings.py before!
    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_full_scaled, dummy_f1)))  # p = 2.9139726704468442e-08, df = 17.516430578284442, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(prec_full_scaled, dummy_pre)))  # p = 8.381146792126513e-12, df = 43.82234134621359, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(rec_full_scaled, dummy_rec)))  # p = 4.4475519123178055e-06, df = 9.741593115383106, p <= (.05, .01) = (True, True)

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_full_scaled, sim_poly_f1)))  # p = 2.232440001999927e-06, df = -10.580722823887339, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(prec_full_scaled, sim_poly_pre)))  # p = 0.004412612421893148, df = -3.770565929289624, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(rec_full_scaled, sim_poly_rec)))  # p = 1.2193668235042846e-05, df = -8.614812648444518, p <= (.05, .01) = (True, True)

    # 1.3 Full feature set scaled from 0 to 1
    # ================================================================================================================

    # create linear kernel SVM
    # prec_full_scaled0, rec_full_scaled0, f1_full_scaled0, cm_full_scaled0 = build_and_evaluate_linearSVC(lc_matrix_full_scaled0, y, [1, 0], "binary")
    # pickle.dump((prec_full_scaled0, rec_full_scaled0, f1_full_scaled0, cm_full_scaled0), open(pdir+'scaled0_full_results.p', "wb"))
    prec_full_scaled0, rec_full_scaled0, f1_full_scaled0, cm_full_scaled_total0 = pickle.load(open(pdir+'scaled0_full_results.p', "rb"))
    cm_full_scaled_total0 = average_fold_performance(cm_full_scaled_total0, 2, 2)

    # report results
    print("Exp. 1.3: Simple SVM with full feature set (scaled 0 to 1)")
    print("Avg. F1: {} [+/- {}]".format(f1_full_scaled0.mean(), f1_full_scaled0.std()))  # 0.6108846975237924 [+/- 0.023273829487588417]
    print("Avg. Recall: {} [+/- {}]".format(rec_full_scaled0.mean(), rec_full_scaled0.std()))  # 0.5966404129307101 [+/- 0.03491028025381015]
    print("Avg. Precision: {} [+/- {}]".format(prec_full_scaled0.mean(), prec_full_scaled0.std()))  # 0.6264000079296094 [+/- 0.011588676437547187]

    # # ================================================================================================================
    # # Experiment 2: compare Hancke 2012, Galasso 2014, Weiß 2015 and full feature set
    # # ================================================================================================================
    #
    # # 2.1 Base feature set scaled [-1,1]
    # # ================================================================================================================
    #
    # # create linear kernel SVM
    # # prec_base_scaled, rec_base_scaled, f1_base_scaled, cm_base_scaled = build_and_evaluate_linearSVC(lc_matrix_base_scaled, y, [1, 0], "binary")
    # # pickle.dump((prec_base_scaled, rec_base_scaled, f1_base_scaled, cm_base_scaled), open(pdir+'scaled_base_results.p', "wb"))
    # prec_base_scaled, rec_base_scaled, f1_base_scaled, cm_base_scaled_total = pickle.load(open(pdir+'scaled_base_results.p', "rb"))
    # cm_base_scaled_total = average_fold_performance(cm_base_scaled_total, 2, 2)
    #
    # # report results
    # print("Exp. 2.1: Simple SVM with base feature set (scaled [-1,1])")
    # print("Avg. F1: {} [+/- {}]".format(f1_base_scaled.mean(), f1_base_scaled.std()))  # 0.6109075733585974 [+/- 0.023268657346735895]
    # print("Avg. Recall: {} [+/- {}]".format(rec_base_scaled.mean(), rec_base_scaled.std()))  # 0.596675147213447 [+/- 0.03496094947870752]
    # print("Avg. Precision: {} [+/- {}]".format(prec_base_scaled.mean(), prec_base_scaled.std()))  # 0.6264135334271344 [+/- 0.011527588896640656]
    #
    # # compare with full
    # print(plot_ttest_results(stats.ttest_rel(f1_full_scaled, f1_base_scaled)))
    # print(plot_ttest_results(stats.ttest_rel(prec_full_scaled, prec_base_scaled)))
    # print(plot_ttest_results(stats.ttest_rel(rec_full_scaled, rec_base_scaled)))
    #
    #
    #
    # # 2.2 Linguistic feature set scaled
    # # ================================================================================================================
    #
    # # create linear kernel SVM
    # # prec_ling_scaled, rec_ling_scaled, f1_ling_scaled, cm_ling_scaled = build_and_evaluate_linearSVC(lc_matrix_ling_scaled, y, [1, 0], "binary")
    # # pickle.dump((prec_ling_scaled, rec_ling_scaled, f1_ling_scaled, cm_ling_scaled), open(pdir+'scaled_ling_results.p', "wb"))
    # prec_ling_scaled, rec_ling_scaled, f1_ling_scaled, cm_ling_scaled_total = pickle.load(open(pdir+'scaled_ling_results.p', "rb"))
    # cm_ling_scaled_total = average_fold_performance(cm_ling_scaled_total, 2, 2)
    #
    # # report results
    # print("Exp. 2.2: Simple SVM with linguistic feature set (scaled)")
    # print("Avg. F1: {} [+/- {}]".format(f1_ling_scaled.mean(), f1_ling_scaled.std()))  # 0.5318456066683487 [+/- 0.012952439216161512]
    # print("Avg. Recall: {} [+/- {}]".format(rec_ling_scaled.mean(), rec_ling_scaled.std()))  # 0.5042460141701635 [+/- 0.021154244472577646]
    # print("Avg. Precision: {} [+/- {}]".format(prec_ling_scaled.mean(), prec_ling_scaled.std()))  # 0.5631469700988248 [+/- 0.005872745150891349]

    # ================================================================================================================
    # Experiment 3: use manual feature selection
    # ================================================================================================================

    # 3.1 Sentential and clausal set
    # ================================================================================================================

    # create linear kernel SVM
    prec_syn_scaled, rec_syn_scaled, f1_syn_scaled, cm_syn_scaled = build_and_evaluate_linearSVC(lc_matrix_sent_scaled, y, [1, 0], "binary")
    pickle.dump((prec_syn_scaled, rec_syn_scaled, f1_syn_scaled, cm_syn_scaled), open(pdir+'scaled_sent_results.p', "wb"))
    prec_syn_scaled, rec_syn_scaled, f1_syn_scaled, cm_syn_scaled_total = pickle.load(open(pdir+'scaled_sent_results.p', "rb"))
    cm_syn_scaled_total = average_fold_performance(cm_syn_scaled, 2, 2)

     # report results
    print("Exp. 2.2: Simple SVM with sentential feature set (scaled)")
    print("Avg. F1: {}".format(f1_syn_scaled.mean()))  # 0.5040611716609863
    print("Avg. Precision: {}".format(prec_syn_scaled.mean()))  # 0.5483667644664996
    print("Avg. Recall: {}".format(rec_syn_scaled.mean()))  # 0.4681555890731026

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_syn_scaled, dummy_f1)))  # p = 0.7305211891583159, df = 0.3553389799503649, p <= (.05, .01) = (False, False)
    print(plot_ttest_results(stats.ttest_rel(prec_syn_scaled, dummy_pre)))  # p = 0.00012612131712169764, df = 6.394257859442024, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(rec_syn_scaled, dummy_rec)))  # p = 0.1747530793425794, df = -1.4733353641748232, p <= (.05, .01) = (False, False)

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_syn_scaled, sim_poly_f1)))  # p = 3.1789875844659405e-07, df = -13.306109775915893, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(prec_syn_scaled, sim_poly_pre)))  # p = 9.7991856924629e-11, df = -33.29416762936627, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(rec_syn_scaled, sim_poly_rec)))  # p = 2.2704426789608512e-06, df = -10.559455532601278, p <= (.05, .01) = (True, True)

    # 3.2 Phrasal set
    # ================================================================================================================

    # create linear kernel SVM
    prec_phr_scaled, rec_phr_scaled, f1_phr_scaled, cm_phr_scaled = build_and_evaluate_linearSVC(lc_matrix_phra_scaled, y, [1, 0], "binary")
    pickle.dump((prec_phr_scaled, rec_phr_scaled, f1_phr_scaled, cm_phr_scaled), open(pdir+'scaled_phr_results.p', "wb"))
    prec_phr_scaled, rec_phr_scaled, f1_phr_scaled, cm_phr_scaled_total = pickle.load(open(pdir+'scaled_phr_results.p', "rb"))
    cm_phr_scaled_total = average_fold_performance(cm_phr_scaled, 2, 2)

     # report results
    print("Exp. 2.2: Simple SVM with phrasal feature set (scaled)")
    print("Avg. F1: {}".format(f1_phr_scaled.mean()))  # 0.5360595299828812
    print("Avg. Precision: {}".format(prec_phr_scaled.mean()))  # 0.5656379644061256
    print("Avg. Recall: {}".format(rec_phr_scaled.mean()))  # 0.5104282519255746

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_phr_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_phr_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_phr_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_phr_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_phr_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_phr_scaled, sim_poly_rec)))

    # 3.2 Lexical set
    # ================================================================================================================

    # create linear kernel SVM
    prec_lex_scaled, rec_lex_scaled, f1_lex_scaled, cm_lex_scaled = build_and_evaluate_linearSVC(lc_matrix_lex_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_lex_scaled.mean()))  # 0.5666647389173642
    print("Avg. Precision: {}".format(prec_lex_scaled.mean()))  # 0.5689223763853553
    print("Avg. Recall: {}".format(rec_lex_scaled.mean()))  # 0.5655868540182112

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_lex_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lex_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lex_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lex_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lex_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lex_scaled, sim_poly_rec)))

    # 3.3 Morphological set
    # ================================================================================================================

    # create linear kernel SVM
    prec_morph_scaled, rec_morph_scaled, f1_morph_scaled, cm_morph_scaled = build_and_evaluate_linearSVC(lc_matrix_morph_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_morph_scaled.mean()))  # 0.5959449701185259
    print("Avg. Precision: {}".format(prec_morph_scaled.mean()))  # 0.6054426621929407
    print("Avg. Recall: {}".format(rec_morph_scaled.mean()))  # 0.5869499643516571

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_morph_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_morph_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_morph_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_morph_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_morph_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_morph_scaled, sim_poly_rec)))

    # 3.3 Cohesion set
    # ================================================================================================================

    # create linear kernel SVM
    prec_coh_scaled, rec_coh_scaled, f1_coh_scaled, cm_coh_scaled = build_and_evaluate_linearSVC(lc_matrix_cohe_scaled, y, [1, 0], "binary")
    print("Avg. Recall: {}".format(f1_coh_scaled.mean()))  # 0.5256685408504247
    print("Avg. F1: {}".format(prec_coh_scaled.mean()))  # 0.6013182906767061
    print("Avg. Precision: {}".format(rec_coh_scaled.mean()))  # 0.46773957408903355

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_coh_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_coh_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_coh_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_coh_scaled, sim_poly_rec)))
    print(plot_ttest_results(stats.ttest_rel(prec_coh_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(rec_coh_scaled, sim_poly_pre)))

    # 3.5 Deagentivation set
    # ================================================================================================================

    # create linear kernel SVM
    prec_deag_scaled, rec_deag_scaled, f1_deag_scaled, cm_deag_scaled = build_and_evaluate_linearSVC(lc_matrix_deag_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_deag_scaled.mean()))  # 0.46038673372917627
    print("Avg. Precision: {}".format(prec_deag_scaled.mean()))  # 0.5021293374777103
    print("Avg. Recall: {}".format(rec_deag_scaled.mean()))  # 0.42591931667161703

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_deag_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_deag_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_deag_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_deag_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_deag_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_deag_scaled, sim_poly_rec)))

    # 3.6 Lexical + Morphological set
    # ================================================================================================================

    # create linear kernel SVM
    prec_lmor_scaled, rec_lmor_scaled, f1_lmor_scaled, cm_lmor_scaled = build_and_evaluate_linearSVC(lc_matrix_lmor_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_lmor_scaled.mean()))  # 0.6026314324899203
    print("Avg. Precision: {}".format(prec_lmor_scaled.mean()))  # 0.6123330267190886
    print("Avg. Recall: {}".format(rec_lmor_scaled.mean()))  # 0.5941219360962805

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_lmor_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmor_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmor_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmor_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmor_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmor_scaled, sim_poly_rec)))

    # 3.7 Lexical + Morphological + Phrasal set
    # ================================================================================================================

    # create linear kernel SVM
    prec_lmorph_scaled, rec_lmorph_scaled, f1_lmorph_scaled, cm_lmorph_scaled = build_and_evaluate_linearSVC(lc_matrix_lmorph_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_lmorph_scaled.mean()))  # 0.6122376813766802
    print("Avg. Precision: {}".format(prec_lmorph_scaled.mean()))  # 0.6253219946460703
    print("Avg. Recall: {}".format(rec_lmorph_scaled.mean()))  # 0.6002875126181014

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_lmorph_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmorph_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmorph_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmorph_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmorph_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmorph_scaled, sim_poly_rec)))

    # 3.7 Lexical + Morphological + Phrasal + Cohesion set
    # ================================================================================================================

    # create linear kernel SVM
    prec_lmpc_scaled, rec_lmpc_scaled, f1_lmpc_scaled, cm_lmpc_scaled = build_and_evaluate_linearSVC(lc_matrix_lmorphcoh_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_lmpc_scaled.mean()))  # 0.6231460170076392
    print("Avg. Precision: {}".format(prec_lmpc_scaled.mean()))  # 0.6428169335923467
    print("Avg. Recall: {}".format(rec_lmpc_scaled.mean()))  # 0.6050985999314726

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_lmpc_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmpc_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmpc_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmpc_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmpc_scaled, sim_poly_pre))) # false!
    print(plot_ttest_results(stats.ttest_rel(rec_lmpc_scaled, sim_poly_rec)))


    # 3.8 Lexical + Morphological + Phrasal + Cohesion + Sen set
    # ================================================================================================================

    # create linear kernel SVM
    prec_lmpcs_scaled, rec_lmpcs_scaled, f1_lmpcs_scaled, cm_lmpcs_scaled = build_and_evaluate_linearSVC(lc_matrix_lmorphcohse_scaled, y, [1, 0], "binary")
    print("Avg. F1: {}".format(f1_lmpcs_scaled.mean()))  # 0.6231460170076392
    print("Avg. Precision: {}".format(prec_lmpcs_scaled.mean()))  # 0.6428169335923467
    print("Avg. Recall: {}".format(rec_lmpcs_scaled.mean()))  # 0.6050985999314726

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_lmpcs_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmpcs_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmpcs_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmpcs_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_lmpcs_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_lmpcs_scaled, sim_poly_rec)))

    # ================================================================================================================
    # Experiment 4: use automatic feature selection
    # ================================================================================================================

    # 4.1 ignore features with no variance
    # ================================================================================================================

    # # create linear kernel SVM
    # selector = VarianceThreshold()
    # mix = selector.fit_transform(lc_matrix_full_scaled)
    #
    # prec_mix, rec_mix, f1_mix, cm_mix = build_and_evaluate_linearSVC(mix, y, [1, 0], "binary")
    # cm_mix = average_fold_performance(cm_mix, 2, 2)
    # pickle.dump((prec_mix, rec_mix, f1_mix, cm_mix), open(pdir+'scaled_mix_results.p', "wb"))
    # prec_mix, rec_mix, f1_mix, cm_mix = pickle.load(open(pdir+'scaled_mix_results.p', "rb"))
    # # results
    # print("Avg. F1: {}".format(f1_mix.mean()))  # 0.6136270620281072
    # print("Avg. Recall: {}".format(rec_mix.mean()))  # 0.5973526165618153
    # print("Avg. Precision: {}".format(prec_mix.mean()))  # 0.6312179177274458

    # 4.2 select features based on previous model
    # ================================================================================================================

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(lc_matrix_full_scaled, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(lc_matrix_full_scaled)
    X_new.shape
    prec_mix, rec_mix, f1_mix, cm_mix = build_and_evaluate_linearSVC(X_new, y, [1, 0], "binary")
    pickle.dump((prec_mix, rec_mix, f1_mix, cm_mix), open(pdir+'scaled_sfm_results.p', "wb"))
    prec_mix, rec_mix, f1_mix, cm_mix = pickle.load(open(pdir+'scaled_sfm_results.p', "rb"))
    cm_mix = average_fold_performance(cm_mix, 2, 2)
    # results
    print("Avg. F1: {}".format(f1_mix.mean()))  # 0.6140107580525791
    print("Avg. Precision: {}".format(prec_mix.mean()))  # 0.6298394178128448
    print("Avg. Recall: {}".format(rec_mix.mean()))  # 0.5995235454332806

    # compare with dummy from other task
    print(plot_ttest_results(stats.ttest_rel(f1_mix, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_mix, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_mix, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_mix, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_mix, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_mix, sim_poly_rec)))