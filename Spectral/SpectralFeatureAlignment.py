import preprocess as pp
import numpy as np


class SpectralFeatureAlignment:
    def __init__(self, nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma):
        self.nclusters = nclusters
        self.nDI = nDI
        self.coocTh = coocTh
        self.sourceFreqTh = sourceFreqTh
        self.targetFreqTh = targetFreqTh
        self.gamma = gamma
        self.U = None
        self.DS = None
        self.DI = None
        self.source = None
        self.target = None

    def spectral_alignment(self, source, target):
        print('spectral initiation')
        self.source, vocab_source = get_features_vocab(source, self.sourceFreqTh, True)
        self.target, vocab_target = get_features_vocab(target, self.targetFreqTh, True)

        vocab = vocab_source.copy()
        for w in vocab_target:
            src = 0
            tar = 0
            if w in vocab_source:
                src = vocab_source[w]
            if w in vocab_target:
                tar = vocab_target[w]
            vocab[w] = src + tar

        corr = {}
        self.correlation(corr, self.source)
        self.correlation(corr, self.target)

        corr = self.threshold(corr, self.coocTh)

        pivots = set(vocab_source.keys()).intersection(set(vocab_target.keys()))

        C = {}
        N = sum(vocab.values())
        for pivot in pivots:
            C[pivot] = 0.0
            for w in vocab_source:
                val = self.getVal(pivot, w, corr)
                C[pivot] += 0.0 if (val < self.coocTh) else self.getPMI(val, vocab[w], vocab[pivot], N)
            for w in vocab_target:
                val = self.getVal(pivot, w, corr)
                C[pivot] += 0.0 if (val < self.coocTh) else self.getPMI(val, vocab[w], vocab[pivot], N)

        pivotList = sorted(C.items(), key=lambda x: x[1], reverse=True)
        DI = {}
        for (w, v) in pivotList[:self.nDI]:
            DI[w] = v

        self.DI = DI
        DS = set(vocab_source.keys()).union(set(vocab_target.keys())) - set(DI)
        DS = list(DS)

        nDS = len(DS)
        nDI = len(DI)

        self.DS = {}
        for i in range(nDS):
            self.DS[DS[i]] = i

        M = np.zeros((nDS, nDI), dtype=np.float64)
        for i in range(nDS):
            for j in range(nDI):
                val = self.getVal(DS[i], list(DI.keys())[j], corr)
                if val > self.coocTh:
                    M[i, j] = val

        nV = len(vocab.keys())

        A = self.affinity_matrix(M, nDS, nV)

        L = self.laplacian_matrix(A, nV)

        U = self.apply_svd(L, self.nclusters)

        self.U = U
        # pp.save(src_string + '_' + tar_string, self)
        print("Spectral alignment is ready")

    @staticmethod
    def threshold(h, t):
        if t == 0:
            return h
        p = {}
        for (key, val) in h.items():
            if val > t:
                p[key] = val
        del h
        return p

    @staticmethod
    def getVal(x, y, M):
        """
        Returns the value of the element (x,y) in M.
        """
        if (x, y) in M.keys():
            return M[(x, y)]
        elif (y, x) in M.keys():
            return M[(y, x)]
        else:
            return 0
        pass

    @staticmethod
    def getPMI(n, x, y, N):
        import math
        """
        Compute the weighted PMI value.
        """
        v = (float(n) * float(N)) / (float(x) * float(y))
        if v <= 0:
            pmi = 0.0
        else:
            pmi = math.log(v)
        res = pmi * (float(n) / float(N))
        return 0 if res < 0 else res

    def functionDS(self, features):
        fea = pp.defaultdict(int)
        feat = list(features.keys())
        x = np.zeros((1, len(self.DS)))
        for f in feat:
            if f in self.DS:
                x[0, self.DS[f]] = features[f]
            else:
                fea[f] = features[f]
        del features

        return x, fea

    @staticmethod
    def correlation(M, data):
        for d in data:
            d = list(d.keys())
            n = len(d)
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (d[i], d[j])
                    rpair = (d[j], d[i])
                    if pair in M:
                        M[pair] += 1
                    elif rpair in M:
                        M[rpair] += 1
                    else:
                        M[pair] = 1

        return M

    @staticmethod
    def affinity_matrix(M, MminusL, nV):
        Mt = M.transpose()
        A = np.zeros((nV, nV), dtype=np.float64)

        for i in range(MminusL, nV):
            for j in range(MminusL):
                A[i, j] = Mt[(i-MminusL), j]

        for i in range(MminusL):
            for j in range(MminusL, nV):
                A[i, j] = M[i, (j-MminusL)]

        return A

    @staticmethod
    def laplacian_matrix(A, nV):
        D = np.zeros((nV, nV), dtype=np.float64)
        for i in range(nV):
            soma = np.sum(A[i, :])
            if soma == 0:
                D[i, i] = soma
            else:
                D[i, i] = 1.0 / np.sqrt(soma)

        L = (D.dot(A)).dot(D)

        return L

    @staticmethod
    def apply_svd(L, k):
        from sparsesvd import sparsesvd
        from scipy.sparse import csc_matrix
        _, eigenvectors = np.linalg.eig(L)
        U, _, _ = sparsesvd(csc_matrix(eigenvectors), k)

        return U.transpose()

    def apply_feature_align(self, x):
        if self.U is not None:
            y = x.dot(self.U[:len(self.DS), :])
            for i in range(self.nclusters):
                y[0, i] = self.gamma * y[0, i]

            return y[0]
        else:
            raise ValueError("U is None, run spectral_alignment before")

    def transform_data(self, dat):
        dt = []
        for features in dat:
            x_line, DI = self.functionDS(features)
            x_line = self.apply_feature_align(x_line)
            dt.append(np.append(np.array(list(DI.values())), x_line))

        dt = pp.pd.DataFrame(dt)
        dt.fillna(0, inplace=True)
        return dt


def get_features_vocab(dt, t, tfidf_bool):
    dicts = pp.get_all_features(dt, tfidf_bool)
    vocab = pp.defaultdict(int)

    for d in dicts:
        for key, val in d.items():
            vocab[key] += val

    vocab = SpectralFeatureAlignment.threshold(vocab, t)

    return dicts, vocab


# if __name__ == "__main__":
#     books_pos, books_neg = dm.pos_neg('C:/Users/igor_/PycharmProjects/Cross-DomainIC/DataSet/books')
#     # dvd_pos, dvd_neg = pp.pos_neg('C:/Users/igor_/PycharmProjects/Cross-DomainIC/DataSet/dvd')
#     el_pos, el_neg = dm.pos_neg('C:/Users/igor_/PycharmProjects/Cross-DomainIC/DataSet/electronics')
#     # kitchen_pos, kitchen_neg = pp.pos_neg('C:/Users/igor_/PycharmProjects/Cross-DomainIC/DataSet/kitchen')
#
#     # labels = [1]*998 + [0]*999
#
#     labels = [1] * 50 + [0] * 50
#     l = books_pos[:50] + books_neg[:50]
#     books = pp.pd.DataFrame(l, columns=['text'])
#
#     # dvd = pp.pd.DataFrame(dvd_pos + dvd_neg, columns=['text'])
#     l = el_pos[:50] + el_neg[:50]
#     electronics = pp.pd.DataFrame(l, columns=['text'])
#
#     # kitchen = pp.pd.DataFrame(kitchen_pos + kitchen_neg, columns=['text'])
#
#     bk_train, bk_test, bk_lb_train, bk_lb_test = train_test_split(books, labels, train_size=0.8,
#                                                                   test_size=0.2, shuffle=True,
#                                                                   stratify=labels)
#
#     el_train, el_test, el_lb_train, el_lb_test = train_test_split(electronics, labels, train_size=0.8,
#                                                                   test_size=0.2, shuffle=True,
#                                                                   stratify=labels)
#     bk_train = bk_train['text']
#     bk_test = bk_test['text']
#     el_train = el_train['text']
#     el_test = el_test['text']
#
#     # nclusters = 50
#     # nDI = 500
#     # coocTh = 10
#     # sourceFreqTh = 10
#     # targetFreqTh = 5
#     # gamma = 1.0
#     # spec = SpectralFeatureAlignment(nclusters, nDI, coocTh, sourceFreqTh, targetFreqTh, gamma)
#     spec = pp.read('Spectral.ig')
#     # spec.spectral_alignment(bk_train, el_train)
#     train = spec.transform_data(spec.source)
#     el_features_test = pp.get_all_features(el_test)
#     test = spec.transform_data(el_features_test)
#     tam = len(train)
#     al = pp.pd.concat([train, test], ignore_index=True)
#     al.fillna(0, inplace=True)
#
#     train = al.iloc[:tam]
#     test = al.iloc[tam:]
#
#     # train = pp.normalize(train)
#     # test = pp.normalize(test)
#     model = clf.choose_model('nb', k_features=len(train.keys()))
#
#     model.fit(train, bk_lb_train)
#     predicts = model.predict(test)
#     acc = accuracy_score(bk_lb_test, predicts)
#     print("Accuracy: " + str(acc * 100) + "%")
#     conf = confusion_matrix(bk_lb_test, predicts)
#     print(conf)
# #     # data = pd.read_csv('DataSet.csv')
# #     # source = SpectralFeatureAlignment.get_domain(data, 'books')
# #     # d2 = SpectralFeatureAlignment.get_domain(data, 'musics')
# #     # d3 = SpectralFeatureAlignment.get_domain(data, 'musical_instruments')
# #     # source = pd.concat([d1,d2,d3], ignore_index=True)
# #     # target = SpectralFeatureAlignment.get_domain(data, 'musics')
# #
# #     # run_grid(data)
# #     # res = SpectralFeatureAlignment.read('result.ig')
# #     # print(res)
# #     run_one(books, movies, clf.choose_model('lr'), 1000, 1000)
