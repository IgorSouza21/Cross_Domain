from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from NeuralNetwork import NeuralNetwork
from timeit import default_timer
import os
import numpy as np
import domains as dm
import preprocess as pp
from sklearn.model_selection import StratifiedKFold
from Spectral.SpectralFeatureAlignment import SpectralFeatureAlignment


model_names = ['knn3', 'knn5', 'knn7', 'wknn3', 'wknn5', 'wknn7', 'nb',
               'dt', 'rf', 'svm', 'lr', 'nn']


def choose_model(str_model, k_features=None):
    if str_model == 'knn3':
        model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=4)
    elif str_model == 'knn5':
        model = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=4)
    elif str_model == 'knn7':
        model = KNeighborsClassifier(n_neighbors=7, metric='euclidean', n_jobs=4)
    elif str_model == 'wknn3':
        model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean', n_jobs=4)
    elif str_model == 'wknn5':
        model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean', n_jobs=4)
    elif str_model == 'wknn7':
        model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean', n_jobs=4)
    elif str_model == 'nb':
        model = GaussianNB()
    elif str_model == 'dt':
        model = DecisionTreeClassifier()
    elif str_model == 'rf':
        model = RandomForestClassifier()
    elif str_model == 'svm':
        model = svm.LinearSVC()
    elif str_model == 'lr':
        model = LogisticRegression(solver='lbfgs', n_jobs=4, max_iter=1000, C=10000)
    elif str_model == 'nn':
        model = NeuralNetwork(k_features, n_hidden_layers=3,
                              n_neurons=[int(k_features/2), int(k_features/5), 20])
    else:
        model = None

    return model


def run_classifier(tuple_classifier, train_dt, train_lb, test_dt, test_lb):
    start = default_timer()
    try:
        tuple_classifier[0].fit(train_dt, train_lb)
    except ValueError:
        train_dt = np.real(train_dt)
        tuple_classifier[0].fit(train_dt, train_lb)

    try:
        predict_label = tuple_classifier[0].predict(test_dt)
    except ValueError:
        test_dt = np.real(test_dt)
        predict_label = tuple_classifier[0].predict(test_dt)

    end = default_timer()
    tuple_classifier[1].append(accuracy_score(test_lb, predict_label))
    tuple_classifier[2].append(end - start)


def classifition_train_test(train_dt, train_lb, test_dt, test_lb, model, k_features=None):
    classifier = [choose_model(model, k_features), [], []]
    run_classifier(classifier, train_dt, train_lb, test_dt, test_lb)

    print('finish classification')
    return classifier[1][0], classifier[2][0]


def train_test(nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target, model,
               nfolds=None):
    if not os.path.isdir('DataSet/%s' % source):
        os.mkdir('DataSet/%s' % source)
    if not os.path.isdir('DataSet/%s' % target):
        os.mkdir('DataSet/%s' % target)

    src_train_dt, src_test_dt, src_train_lb, src_test_lb = dm.return_domain(source)
    tar_train_dt, tar_test_dt, tar_train_lb, tar_test_lb = dm.return_domain(target)

    spec = SpectralFeatureAlignment(nclusters, nDI, coocTh, sourceFreqTh,
                                    targetFreqTh, gamma)
    spec.spectral_alignment(source, target, src_train_dt, tar_train_dt)

    train = spec.transform_data(spec.source)

    tar_feat = []
    for a in tar_test_dt:
        tar_feat.append(pp.get_features(a))

    test = spec.transform_data(tar_feat)

    tam = len(train)
    al = pp.pd.concat([train, test], ignore_index=True)
    al.fillna(0, inplace=True)
    train = al.iloc[:tam]
    test = al.iloc[tam:]

    acc, time = classifition_train_test(train, src_train_lb, test,
                                        tar_test_lb, model,
                                        len(train.keys()))

    return acc, time


def kfold(nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target, model, nfolds):
    seeds = open('seeds.txt', 'r')
    kf = StratifiedKFold(nfolds, random_state=int(seeds.read()))

    if not os.path.isdir('DataSet/%s' % source):
        os.mkdir('DataSet/%s' % source)
    if not os.path.isdir('DataSet/%s' % target):
        os.mkdir('DataSet/%s' % target)

    src_pos, src_neg = dm.pos_neg('DataSet/' + source)
    tar_pos, tar_neg = dm.pos_neg('DataSet/' + target)

    # n = 100
    # src_pos = src_pos[:n]
    # src_neg = src_neg[:n]
    # tar_pos = tar_pos[:n]
    # tar_neg = tar_neg[:n]

    labels = [1] * len(src_pos) + [0] * len(src_neg)

    source_data = src_pos + src_neg
    target_data = tar_pos + tar_neg

    source_index = []
    target_index = []
    for dt_train, dt_test in kf.split(source_data, labels):
        source_index.append((dt_train, dt_test))
    for dt_train, dt_test in kf.split(target_data, labels):
        target_index.append((dt_train, dt_test))

    accs = []
    times = []

    for i in range(nfolds):
        print('fold ' + str(i+1))
        spec = SpectralFeatureAlignment(nclusters, nDI, coocTh, sourceFreqTh,
                                        targetFreqTh, gamma)
        train_dt_src = [source_data[a] for a in source_index[i][0]]
        train_dt_tar = [target_data[a] for a in target_index[i][0]]
        test_dt_tar = [target_data[a] for a in target_index[i][1]]
        spec.spectral_alignment(source, target, train_dt_src, train_dt_tar)

        train = spec.transform_data(spec.source)

        tar_feat = []
        for a in test_dt_tar:
            tar_feat.append(pp.get_features(a))

        test = spec.transform_data(tar_feat)

        tam = len(train)
        al = pp.pd.concat([train, test], ignore_index=True)
        al.fillna(0, inplace=True)
        train = al.iloc[:tam]
        test = al.iloc[tam:]

        acc, time = classifition_train_test(train, [labels[a] for a in source_index[i][0]], test,
                                            [labels[a] for a in target_index[i][1]], model,
                                            len(train.keys()))
        accs.append(acc)
        times.append(time)

    return (np.mean(accs), np.median(accs), np.std(accs)), np.median(times)
