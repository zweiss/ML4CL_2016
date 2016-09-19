__author__ = 'zweiss'

import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, RFECV, VarianceThreshold
from data_extraction import *
import pickle
from svc import *
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from scipy import stats

if __name__ == '__main__':


    # set working directory ### CHANGE ###
    os.chdir('/Volumes/INTENSO/ml-project/ml-analysis')
    # set seed for reproducibility
    seed = 4050712
    np.random.seed(seed)
    pdir = './rsrc/party-complexity/'

    # get complexity table containing a balanced set of speeches from each party
    lc_table_raw = pd.read_csv("./rsrc/bundesparser-160903_2_meta-new.csv", low_memory=False)
    sample_size = 11059
    lc_table = lc_table_raw[lc_table_raw['party'] == 'cdu'].sample(n=sample_size)
    lc_table = lc_table.append(lc_table_raw[lc_table_raw['party'] == 'fdp'].sample(n=sample_size))
    lc_table = lc_table.append(lc_table_raw[lc_table_raw['party'] == 'gruene'].sample(n=sample_size))
    lc_table = lc_table.append(lc_table_raw[lc_table_raw['party'] == 'linke'].sample(n=sample_size))
    lc_table = lc_table.append(lc_table_raw[lc_table_raw['party'] == 'spd'].sample(n=sample_size))

    # get response variable
    y_cat = ['cdu'] * sample_size + ['fdp'] * sample_size + ['gruene'] * sample_size + ['linke'] * sample_size + ['spd'] * sample_size
    lb = LabelEncoder()  # change label encoding
    y = lb.fit_transform(y_cat)

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


    # 0. Dummy classifier
    # ================================================================================================================

    # create an SVM with a linear kernel
    dummy_pre, dummy_rec, dummy_f1, dummy_cm = build_and_evaluate_dummy_classifier(np.zeros((len(y),2)), y, [0, 1, 2, 3, 4], "weighted", strategy='stratified')
    cm_dummy = average_fold_performance(dummy_cm, 5, 5)

    #  get features and report results
    print("Exp. 1.3: Dummy")
    print("Avg. F1: {}".format(dummy_f1.mean()))            # 0.2003895695710367
    print("Avg. Recall: {}".format(dummy_rec.mean()))       # 0.20046988454583392
    print("Avg. Precision: {}".format(dummy_pre.mean()))    # 0.2003953658546916
    print("Avg. CM: {}".format(cm_dummy))

    sim_poly_pre, sim_poly_rec, sim_poly_f1, cm_poly = pickle.load(open('./rsrc/party-embeddings/party_sim_poly_results.p', "rb"))



    # ================================================================================================================
    # Experiment 0: compare scaling
    # ================================================================================================================

    # 0.1 Full feature set not scaled
    # ================================================================================================================

    # create linear kernel SVM
    # prec_full, rec_full, f1_full, cm_full = build_and_evaluate_linearSVC(lc_matrix_full, y, [0, 1, 2, 3, 4], "weighted")
    # cm_full_total = average_fold_performance(cm_full, 5, 5)
    # pickle.dump((prec_full, rec_full, f1_full, cm_full_total), open(pdir+'sim_full_results.p', "wb"))
    prec_full, rec_full, f1_full, cm_full_total = pickle.load(open(pdir+'sim_full_results.p', "rb"))

    # report results
    print("Exp. 1.1: Simple SVM with full feature set (not scaled)")
    print("Avg. F1: {}".format(f1_full.mean()))  # 0.22436787611393133
    print("Avg. Recall: {}".format(rec_full.mean()))  # 0.24702080793368952
    print("Avg. Precision: {}".format(prec_full.mean()))  # 0.24012527144013013


    # 0.2 Full feature set scaled from 0 to 1
    # ================================================================================================================

    # create linear kernel SVM
    # prec_full_scaled0, rec_full_scaled0, f1_full_scaled0, cm_full_scaled0 = build_and_evaluate_linearSVC(lc_matrix_full_scaled0, y, [0, 1, 2, 3, 4], "weighted")
    # cm_full_scaled_total0 = average_fold_performance(cm_full_scaled0, 5, 5)
    # pickle.dump((prec_full_scaled0, rec_full_scaled0, f1_full_scaled0, cm_full_scaled_total0), open(pdir+'scaled0_full_results.p', "wb"))
    prec_full_scaled0, rec_full_scaled0, f1_full_scaled0, cm_full_scaled_total0 = pickle.load(open(pdir+'scaled0_full_results.p', "rb"))

    # report results
    print("Exp. 1.3: Simple SVM with full feature set (scaled 0 to 1)")
    print("Avg. F1: {}".format(f1_full_scaled0.mean()))  # 0.28303040183463535
    print("Avg. Recall: {}".format(rec_full_scaled0.mean()))  # 0.2940223544140149
    print("Avg. Precision: {}".format(prec_full_scaled0.mean()))  # 0.2882804426776272

    # ================================================================================================================
    # Experiment 1: compare combined feature sets
    # ================================================================================================================

    # 1.3 full features
    # ================================================================================================================

    # create linear kernel SVM
    # prec_full_scaled, rec_full_scaled, f1_full_scaled, cm_full_scaled = build_and_evaluate_linearSVC(lc_matrix_full_scaled, y, [0, 1, 2, 3, 4], "weighted")
    # pickle.dump((prec_full_scaled, rec_full_scaled, f1_full_scaled, cm_full_scaled), open(pdir+'party_full_results.p', "wb"))
    prec_full_scaled, rec_full_scaled, f1_full_scaled, cm_full_scaled = pickle.load(open(pdir+'party_full_results.p', "rb"))
    cm_full_scaled = average_fold_performance(cm_full_scaled, 5, 5)

    # report results
    print("Exp. 1.3: Simple SVM with full feature set (scaled)")
    print("Avg. F1: {}".format(f1_full_scaled.mean()))  # 0.2773317341181111
    print("Avg. Recall: {}".format(rec_full_scaled.mean()))  # 0.289627977383748
    print("Avg. Precision: {}".format(prec_full_scaled.mean()))  # 0.2834474382639513


    # run ml_experiment-party-embeddings.py before!
    # compare with dummy from other task

    print(plot_ttest_results(stats.ttest_rel(f1_full_scaled, dummy_f1)))  # p = 2.9139726704468442e-08, df = 17.516430578284442, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(prec_full_scaled, dummy_pre)))  # p = 8.381146792126513e-12, df = 43.82234134621359, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(rec_full_scaled, dummy_rec)))  # p = 4.4475519123178055e-06, df = 9.741593115383106, p <= (.05, .01) = (True, True)

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_full_scaled, sim_poly_f1)))  # p = 2.232440001999927e-06, df = -10.580722823887339, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(prec_full_scaled, sim_poly_pre)))  # p = 0.004412612421893148, df = -3.770565929289624, p <= (.05, .01) = (True, True)
    print(plot_ttest_results(stats.ttest_rel(rec_full_scaled, sim_poly_rec)))  # p = 1.2193668235042846e-05, df = -8.614812648444518, p <= (.05, .01) = (True, True)



    # ================================================================================================================
    # Experiment 2: compare single feature sets
    # ================================================================================================================

    # 2.1 sentence features
    # ================================================================================================================

    # create linear kernel SVM
    prec_syn_scaled, rec_syn_scaled, f1_syn_scaled, cm_syn_scaled = build_and_evaluate_linearSVC(lc_matrix_sent_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_syn_scaled.mean())  # 0.22569467048
    print("Avg. Precision: {}".format(prec_syn_scaled.mean()))  # 0.23627752704097094
    print("Avg. Recall: {}".format(rec_syn_scaled.mean()))  # 0.23937055796028245

    print(plot_ttest_results(stats.ttest_rel(f1_syn_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_syn_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_syn_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_syn_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_syn_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_syn_scaled, sim_poly_rec)))

    # 2.2 phrasal features
    # ================================================================================================================

    # create linear kernel SVM
    prec_phr_scaled, rec_phr_scaled, f1_phr_scaled, cm_phr_scaled = build_and_evaluate_linearSVC(lc_matrix_phra_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_phr_scaled.mean())  # 0.246266792379
    print("Avg. Precision: {}".format(prec_phr_scaled.mean()))  # 0.2544310060281996
    print("Avg. Recall: {}".format(rec_phr_scaled.mean()))  # 0.2611088836703133

    print(plot_ttest_results(stats.ttest_rel(f1_phr_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_phr_scaled, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_phr_scaled, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_phr_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec_phr_scaled, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec_phr_scaled, sim_poly_rec)))

    # 2.3 lex features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_lex_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_lex_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_lex_scaled.mean())  # 0.243025420554
    print("Avg. Precision: {}".format(prec.mean()))  # 0.2508692875432403
    print("Avg. Recall: {}".format(rec.mean()))  # 0.2586120789114087

    print(plot_ttest_results(stats.ttest_rel(f1_lex_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lex_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.4 morph features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_morph_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_morph_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_morph_scaled.mean())  # 0.245130859245
    print("Avg. Precision: {}".format(prec.mean()))  # 0.2535870378559001
    print("Avg. Recall: {}".format(rec.mean()))  # 0.26045734905451956

    print(plot_ttest_results(stats.ttest_rel(f1_morph_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_morph_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.5 cohesion features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_cohe_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_cohe_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_cohe_scaled.mean())  # 0.232734233398
    print("Avg. Precision: {}".format(prec.mean()))  # 0.24535086241382298
    print("Avg. Recall: {}".format(rec.mean()))  # 0.25061901761678385

    print(plot_ttest_results(stats.ttest_rel(f1_cohe_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_cohe_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.6 deag features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_deag_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_deag_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_deag_scaled.mean())  # 0.208630890212
    print("Avg. Precision: {}".format(prec.mean()))  # 0.22225892620905005
    print("Avg. Recall: {}".format(rec.mean()))  # 0.22363666713033803

    print(plot_ttest_results(stats.ttest_rel(f1_deag_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_deag_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.7 lex + morph features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_lmor_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_lmor_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_lmor_scaled.mean())  # 0.264462378563
    print("Avg. Precision: {}".format(prec.mean()))  # 0.2717344609186696
    print("Avg. Recall: {}".format(rec.mean()))  # 0.2776019736034628

    print(plot_ttest_results(stats.ttest_rel(f1_lmor_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmor_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.8 lex + morph + phrasal features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_lmorph_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_lmorph_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_lmorph_scaled.mean())  # 0.273761301955
    print("Avg. Precision: {}".format(prec.mean()))  # 0.2810739936196958
    print("Avg. Recall: {}".format(rec.mean()))  # 0.2869518627314606

    print(plot_ttest_results(stats.ttest_rel(f1_lmorph_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmorph_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.9 lex + morph + phrasal + coh features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_lmorphcoh_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_lmorphcoh_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_lmorphcoh_scaled.mean())  # 0.277982734716
    print("Avg. Precision: {}".format(prec.mean()))  # 0.284116669141534
    print("Avg. Recall: {}".format(rec.mean()))  # 0.2900802042335922

    print(plot_ttest_results(stats.ttest_rel(f1_lmorphcoh_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmorphcoh_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # 2.10 lex + morph + phrasal + coh + sent features
    # ================================================================================================================

    # create linear kernel SVM
    prec,rec, f1_lmorphcohse_scaled, _ = build_and_evaluate_linearSVC(lc_matrix_lmorphcohse_scaled, y, [0, 1, 2, 3, 4], "weighted")
    print(f1_lmorphcohse_scaled.mean())  # 0.28006723975
    print("Avg. Precision: {}".format(prec.mean()))  # 0.2858843548106197
    print("Avg. Recall: {}".format(rec.mean()))  # 0.29196103524175

    print(plot_ttest_results(stats.ttest_rel(f1_lmorphcohse_scaled, dummy_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, dummy_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, dummy_rec)))

    # compare with embeddings
    print(plot_ttest_results(stats.ttest_rel(f1_lmorphcohse_scaled, sim_poly_f1)))
    print(plot_ttest_results(stats.ttest_rel(prec, sim_poly_pre)))
    print(plot_ttest_results(stats.ttest_rel(rec, sim_poly_rec)))

    # ================================================================================================================
    # Experiment 3: use automatic feature selection
    # ================================================================================================================

    # 3.1 ignore features with no variance
    # ================================================================================================================

    # # create linear kernel SVM
    # selector = VarianceThreshold()
    # mix = selector.fit_transform(lc_matrix_full_scaled)
    #
    # prec_mix, rec_mix, f1_mix, cm_mix = build_and_evaluate_linearSVC(mix, y, [0, 1, 2, 3, 4], "weighted")
    # cm_mix = average_fold_performance(cm_mix, 2, 2)
    # pickle.dump((prec_mix, rec_mix, f1_mix, cm_mix), open(pdir+'scaled_mix_results.p', "wb"))
    # prec_mix, rec_mix, f1_mix, cm_mix = pickle.load(open(pdir+'scaled_mix_results.p', "rb"))
    # # results
    # print("Avg. F1: {}".format(f1_mix.mean()))  # 0.6136270620281072
    # print("Avg. Recall: {}".format(rec_mix.mean()))  # 0.5973526165618153
    # print("Avg. Precision: {}".format(prec_mix.mean()))  # 0.6312179177274458

    # 3.2 select features based on previous model
    # ================================================================================================================

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(lc_matrix_full_scaled, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(lc_matrix_full_scaled)
    X_new.shape
    prec_mix, rec_mix, f1_mix, cm_mix = build_and_evaluate_linearSVC(X_new, y, [0, 1, 2, 3, 4], "weighted")
    cm_mix = average_fold_performance(cm_mix, 5, 5)
    pickle.dump((prec_mix, rec_mix, f1_mix, cm_mix), open(pdir+'scaled_sfm_results.p', "wb"))
    prec_mix, rec_mix, f1_mix, cm_mix = pickle.load(open(pdir+'scaled_sfm_results.p', "rb"))
    # results
    print("Avg. F1: {}".format(f1_mix.mean()))  # 0.26550903234989454
    print("Avg. Precision: {}".format(prec_mix.mean()))  # 0.27446214347765263
    print("Avg. Recall: {}".format(rec_mix.mean()))  # 0.2804594273931579