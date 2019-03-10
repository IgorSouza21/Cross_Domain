import preprocess as pp
import numpy as np
import scipy as sp


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

    def spectral_alignment(self, str_src, str_tar, source, target):
        self.source = self.get_features(source, str_src)
        self.get_features(target, str_tar)

        s = {}
        self.get_vocab(s, str_src)
        s = self.threshold(s, self.sourceFreqTh)

        t = {}
        self.get_vocab(t, str_tar)
        t = self.threshold(t, self.targetFreqTh)

        v = s.copy()
        for w in t:
            v[w] = s.get(w, 0) + t[w]

        m = {}
        self.correlation(v, m, str_src)
        self.correlation(v, m, str_tar)

        m = self.threshold(m, self.coocTh)

        pivots = set(s.keys()).intersection(set(t.keys()))

        C = {}
        N = sum(v.values())
        for pivot in pivots:
            C[pivot] = 0.0
            for w in s:
                val = self.getVal(pivot, w, m)
                C[pivot] += 0.0 if (val < self.coocTh) else self.getPMI(val, v[w], v[pivot], N)
            for w in t:
                val = self.getVal(pivot, w, m)
                C[pivot] += 0.0 if (val < self.coocTh) else self.getPMI(val, v[w], v[pivot], N)

        pivotList = sorted(C.items(), key=lambda x: x[1], reverse=True)
        DI = {}
        for (w, x) in pivotList[:self.nDI]:
            DI[w] = x

        self.DI = DI
        DS = set(s.keys()).union(set(t.keys())) - set(DI)
        DS = list(DS)

        nDS = len(DS)
        nDI = len(DI)

        self.DS = {}
        for i in range(nDS):
            self.DS[DS[i]] = i

        M = np.zeros((nDS, nDI), dtype=np.float64)
        for i in range(nDS):
            for j in range(nDI):
                val = self.getVal(DS[i], list(DI.keys())[j], m)
                if val > self.coocTh:
                    M[i, j] = val

        nV = len(v.keys())

        A = self.affinity_matrix(M, nDS, nV)

        L = self.laplacian_matrix(A, nV)

        U = self.apply_svd(L, self.nclusters)

        self.U = U
        # pp.save(src_string + '_' + tar_string, self)
        print('Spectral alignment is ready.')

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
    def correlation(v, m, domain_name):
        file = open('DataSet/%s/features.txt' % domain_name)
        for f in file:
            line = f.strip().split()
            p = []
            for w in line:
                if w in v:
                    p.append(w)
            n = len(p)
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (p[i], p[j])
                    rpair = (p[j], p[i])
                    if pair in m:
                        m[pair] += 1
                    elif rpair in m:
                        m[rpair] += 1
                    else:
                        m[pair] = 1
        file.close()

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

        _, eigenvectors = sp.linalg.eig(L)
        eigenvectors = np.real(eigenvectors)
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

    def get_features(self, dt, str_domain):
        dicts = []
        file = open("DataSet/%s/features.txt" % str_domain, 'w')
        for d in dt:
            feat = pp.get_features(d)
            for f in feat.items():
                for i in range(f[1]):
                    file.write('%s ' % f[0])
            dicts.append(feat)
            file.write('\n')
        file.close()

        return dicts

    def get_vocab(self, S, fname):
        f = open('DataSet/%s/features.txt' % fname)
        for line in f:
            p = line.strip().split()
            for w in p:
                S[w] = S.get(w, 0) + 1
        f.close()
