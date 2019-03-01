import preprocess as pp
import numpy as np


def calculate_corr(M, d):
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
        try:
            U, _, _ = sparsesvd(csc_matrix(eigenvectors), k)
        except(ValueError, IndexError):
            return eigenvectors[:k].transpose()

        return U.transpose()

    def apply_feature_align(self, x):
        if self.U is not None:
            tam = len(self.DS)
            y = x.dot(self.U[:tam, :])
            for i in range(len(y[0])):
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
