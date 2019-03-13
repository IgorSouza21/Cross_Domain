import classification as clf
import preprocess as pp


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

    def worker(self, function, param):
        nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target, model = param

        acc, time = function(nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma, source, target,
                             model, self.nfolds)

        self.all_results[param] = (acc, time)
        if acc > self.best_acc:
            self.best_acc = acc
            self.best = param

    def search(self, evaluation=None):
        params = self.gerar_parametros()
        tam = len(params)

        for param in params:
            print('iteration %d' % tam)
            if evaluation is None:
                self.worker(clf.train_test, param)
            elif evaluation == 'tt':
                self.worker(clf.train_test, param)
            elif evaluation == 'kf':
                if self.nfolds is None:
                    raise AttributeError('attribute nfolds not specified')
                self.worker(clf.kfold, param)

            tam -= 1


def run_grid(model, source, target, eval=None, nfold=None):
    # parameters = {'nclusters': [2, 5, 7],
    #               'nDI': [60, 80, 100],
    #               'coocTh': [5, 10, 15],
    #               'sourceFreqTh': [5, 10, 15],
    #               'targetFreqTh': [5, 10, 15],
    #               'gamma': [0.1, 0.5, 1.0],
    #               'source': ['books', 'dvd', 'electronics', 'kitchen'],
    #               'target': ['books', 'dvd', 'electronics', 'kitchen'],
    #               'model': model}

    parameters = {'nclusters': [100],
                  'nDI': [500],
                  'coocTh': [2, 5, 10, 15],
                  'sourceFreqTh': [2, 5, 10, 15],
                  'targetFreqTh': [2, 5, 10, 15],
                  'gamma': [0.6],
                  'source': [source],
                  'target': [target],
                  'model': [model]}

    grid = GridSearchSpectral(parameters, nfold)
    grid.search(eval)
    pp.save_pickle('Spectral/results/%s-%s_%s.rs' % (source, target, model), grid)
    print("Best Result Found")
    print(grid.best)
    print(grid.best_acc)
    print('====================FINISHED=====================')
