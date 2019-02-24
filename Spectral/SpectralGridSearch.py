import classification as clf
from Spectral.SpectralFeatureAlignment import SpectralFeatureAlignment
import domains as dm
import preprocess as pp
# import multiprocessing as mp


class GridSearchSpectral:
    def __init__(self, parameters):
        self.parameters = parameters
        self.best = None
        self.best_acc = 0.0
        self.all_results = {}

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

    def worker(self, param):
        nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target, model = param
        src_train_dt, src_test_dt, src_train_lb, src_test_lb = dm.return_domain(source)
        tar_train_dt, tar_test_dt, tar_train_lb, tar_test_lb = dm.return_domain(target)

        spec = SpectralFeatureAlignment(nclusters, nDI, coocTh, sourceFreqTh,
                                        targetFreqTh, gamma)
        spec.spectral_alignment(src_train_dt, tar_train_dt)

        train = spec.transform_data(spec.source)

        tar_feat = pp.get_all_features(tar_test_dt, True)
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

    def search(self):
        params = self.gerar_parametros()

        length = len(params)

        for param in params:
            print("iteration " + str(length))
            self.worker(param)
            length -= 1
        # processes = [mp.Process(target=self.worker, args=(param,)) for param in params]
        #
        # for p in processes:
        #     p.start()
        #
        # for p in processes:
        #     p.join()

        # pool = mp.Pool(int(mp.cpu_count()/2))
        # results = [pool.apply_async(self.worker, args=(param,)) for param in params]
        # [a.get() for a in results]




