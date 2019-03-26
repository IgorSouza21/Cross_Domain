import preprocess as pp
# import numpy as np


class SensitiveThesaurus:
    def __init__(self):
        self.source = None

    def sensitive_thesaurus(self, str_src, str_tar, source, lb_source, target):
        self.source = self.get_features(source, str_src, lb_source)
        self.get_features(target, str_tar)

    def get_features(self, dt, str_domain, lb=None):
        dicts = []
        file = open("DataSet/%s/features.txt" % str_domain, 'w')
        for i in range(len(dt)):
            feat = pp.get_features(dt[i])
            for f in feat.items():
                for j in range(f[1]):
                    if lb[i] == 1:
                        file.write('%s*P ' % f[0])
                    else:
                        file.write('%s*N ' % f[0])

            dicts.append(feat)
            file.write('\n')
        file.close()

        return dicts
