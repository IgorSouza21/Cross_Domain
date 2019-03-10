import classification as clf
from sklearn.model_selection import StratifiedKFold
from Spectral.SpectralFeatureAlignment import SpectralFeatureAlignment
import domains as dm
import preprocess as pp
import numpy as np
import os


class GridSearchSpectral:
    def __init__(self, parameters, nfolds):
        self.parameters = parameters
        self.best = None
        self.best_acc = 0.0
        self.all_results = {}
        if nfolds is None:
            self.nfolds = 5
        else:
            self.nfolds = nfolds

    def gerar_parametros(self):
        print('generating parameters')
        l1 = self.parameters['nclusters']
        l2 = self.parameters['nDI']
        l3 = self.parameters['coocTh']
        l4 = self.parameters["sourceFreqTh"]
        l5 = self.parameters["targetFreqTh"]
        l6 = self.parameters["gamma"]
        l7 = self.parameters['source']
        l8 = self.parameters['target']
        l9 = self.parameters['model']

        z = []
        for a in l1:
            for b in l2:
                for c in l3:
                    for d in l4:
                        for e in l5:
                            for f in l6:
                                for g in l7:
                                    for h in l8:
                                        for i in l9:
                                            z.append((a, b, c, d, e, f, g, h, i))

        return z

    def train_test(self, param):
        nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target, model = param

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

        acc, time = clf.train_test_one_model(train, src_train_lb, test,
                                             tar_test_lb, model,
                                             len(train.keys()))

        self.all_results[param] = (acc, time)
        if acc > self.best_acc:
            self.best_acc = acc
            self.best = param

    def kfold(self, param):
        kfold = StratifiedKFold(self.nfolds)
        nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target, model = param

        if not os.path.isdir('DataSet/%s' % source):
            os.mkdir('DataSet/%s' % source)
        if not os.path.isdir('DataSet/%s' % target):
            os.mkdir('DataSet/%s' % target)

        src_pos, src_neg = dm.pos_neg('DataSet/' + source)
        tar_pos, tar_neg = dm.pos_neg('DataSet/' + target)

        # n = 50
        # src_pos = src_pos[:n]
        # src_neg = src_neg[:n]
        # tar_pos = tar_pos[:n]
        # tar_neg = tar_neg[:n]

        labels = [1] * len(src_pos) + [0] * len(src_neg)

        source_data = src_pos + src_neg
        target_data = tar_pos + tar_neg

        source_index = []
        target_index = []
        for dt_train, dt_test in kfold.split(source_data, labels):
            source_index.append((dt_train, dt_test))
        for dt_train, dt_test in kfold.split(target_data, labels):
            target_index.append((dt_train, dt_test))

        accs = []
        times = []
        for i in range(self.nfolds):
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

            acc, time = clf.train_test_one_model(train, [labels[a] for a in source_index[i][0]], test,
                                                 [labels[a] for a in target_index[i][1]], model,
                                                 len(train.keys()))
            accs.append(acc)
            times.append(time)

        mean_acc = np.mean(accs)
        median_acc = np.median(accs)
        std_acc = np.std(accs)
        print('Accuracy : ', mean_acc)
        print('Median : ', median_acc)
        print('Standard Deviation : ', std_acc)
        # print('Max : ', np.max(accs))
        # print('Min : ', np.min(accs))
        self.all_results[param] = (mean_acc, median_acc, std_acc, np.median(times))
        if mean_acc > self.best_acc:
            self.best_acc = mean_acc
            self.best = param

    def worker(self, function, param):
        function(param)

    def search(self, evaluation=None):
        params = self.gerar_parametros()
        tam = len(params)

        for param in params:
            print('iteration %d' % tam)
            if evaluation is None:
                self.worker(self.train_test, param)
            elif evaluation == 'tt':
                self.worker(self.train_test, param)
            elif evaluation == 'kf':
                self.worker(self.kfold, param)

            tam -= 1

