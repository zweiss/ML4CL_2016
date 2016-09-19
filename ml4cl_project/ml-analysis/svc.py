__author__ = 'zweiss'

import numpy as np
from sklearn import svm, metrics
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

# =====================================================================================================================
# Linear SVMs
# =====================================================================================================================


def build_and_evaluate_svc(X, y, labels, average, num_folds=10, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                           tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                           decision_function_shape=None, random_state=None, multi_class="ovr"):

    model = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                    probability=False, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                    max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)

    return build_and_evaluate_some_model(X, y, labels, model, average, num_folds)


def build_and_evaluate_linearSVC(X, y, labels, average, num_folds=10, penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0,
                           fit_intercept=True, intercept_scaling=1, class_weight=None, multi_class="ovr"):

    model = svm.LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, class_weight=class_weight, multi_class=multi_class)

    return build_and_evaluate_some_model(X, y, labels, model, average, num_folds)


def build_and_evaluate_linearLogReg(X, y, labels, average, num_folds=10, penalty='l2', solver='sag',
                                    dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                                    class_weight=None, multi_class="ovr"):

    model = svm.linear.LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                          intercept_scaling=intercept_scaling, class_weight=class_weight,
                                          solver=solver, multi_class=multi_class, n_jobs=1)

    return build_and_evaluate_some_model(X, y, labels, model, average, num_folds)


def build_and_evaluate_dummy_classifier(X, y, labels, average, num_fols=10, strategy='stratified'):

    model = DummyClassifier(strategy=strategy)

    return build_and_evaluate_some_model(X, y, labels, model, average, num_fols)


def build_and_evaluate_some_model(X, y, labels, model, average, num_folds=10):

    # get stratified folds
    skf = StratifiedKFold(y, num_folds)

    # test and train through folds
    n_categories = len(np.unique(y))
    cm = []
    # f1 = np.zeros(n_categories ** 2)
    # precision = np.zeros(n_categories ** 2)
    # recall = np.zeros(n_categories ** 2)
    f1 = []
    precision = []
    recall = []
    for i, (i_train, i_test) in enumerate(skf):
        print("Started fold {}/{} ...".format(i+1, num_folds))
        model.fit(X[i_train], y[i_train])  # fit model
        y_pred = model.predict(X[i_test])  # predict test data
        y_true = y[i_test]
        # get metrics
        cm.append(metrics.confusion_matrix(y_true, y_pred, labels))
        precision.append(metrics.precision_score(y_true, y_pred, average=average))
        recall.append(metrics.recall_score(y_true, y_pred, average=average))
        f1.append(metrics.f1_score(y_true, y_pred, average=average))

    return np.array(precision), np.array(recall), np.array(f1), np.array(cm)


# =====================================================================================================================
# Helper
# =====================================================================================================================


def plot_confusion_matrix(target_names, cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def average_fold_performance(fold_results, r, c):

    avg = np.zeros((r,c))
    for result in fold_results:
        avg += result
    avg /= len(fold_results)

    return avg


def plot_ttest_results(ttest_result):
    p = ttest_result[1]
    df = ttest_result[0]
    return "p = {}, df = {}, p <= (.05, .01) = ({}, {})".format(p, df, p <= .05, p <= .01)
