import pandas as pd
from PSOFeatureSelection import PsoFeatureSelection
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing
from sklearn import feature_selection as fs
import pickle
import nltk
from collections import defaultdict
import math
# import multiprocessing as mp


def transform_in_dataframe(values, columns):
    return pd.DataFrame([list(i) for i in zip(*values)], columns=columns)


def tokenize(t):
    import string
    from nltk.corpus import stopwords
    from collections import Counter

    stop = stopwords.words('english')
    x = []
    t = word_tokenize(t)
    for a in t:
        a = a.lower()
        count = Counter(a)
        if a not in stop and (a is not 'no' or a is not 'not' or a is not 'nor') and a not in string.punctuation and \
                a.isalpha() and len(a) > 2 and len(count) > 1:
            x.append(WordNetLemmatizer().lemmatize(a))

    return x


def break_data_label(base):
    columns = list(base.columns)
    label = columns.pop()
    b = base.copy()
    labels = b.pop(label)

    return b, labels


def pre_process(data_set, select_type, k, model):
    sel = None
    data, labels = break_data_label(data_set)
    data = data['text']
    selector = []
    if select_type is 'mutual':
        selector = fs.SelectKBest(fs.mutual_info_classif, k=k)
    elif select_type is 'chi':
        selector = fs.SelectKBest(fs.chi2, k=k)
    elif select_type is 'anova':
        selector = fs.SelectKBest(fs.f_classif, k=k)
    elif select_type is 'pso':
        selector = PsoFeatureSelection(10, model)
        sel = 'pso'

    matrix = []
    for d in data:
        matrix.append(get_features(d))
    matrix = pd.DataFrame(matrix)
    if sel is 'pso':
        selector.dimension = len(matrix.columns)
    new_data = selector.fit_transform(matrix, labels)
    features = []
    r = list(selector.get_support())
    # lb = [a for a in extractor.vocabulary_.keys()]
    lb = [a for a in matrix.keys()]
    for i in range(len(r)):
        if r[i]:
            features.append(lb[i])
    # dat = pd.DataFrame(new_data.toarray(), columns=features)
    dat = pd.DataFrame(new_data, columns=features)
    lb = pd.DataFrame()
    lb['class_label'] = labels
    return dat, lb


def normalize(base):
    cols = list(base.columns)
    if cols[-1] == 'label' or cols[-1] == 'class_label':
        label = cols.pop()
        new_base = preprocessing.normalize(base[cols])
        labels = pd.DataFrame(base[label], columns=[label])
        new_base = pd.DataFrame(new_base, columns=cols)
        return new_base.join(labels)
    else:
        new_base = preprocessing.normalize(base[cols])
        new_base = pd.DataFrame(new_base, columns=cols)
        return new_base


def filter_by_pos(x):
    if 'NN' in x[1] or 'VB' in x[1] or 'JJ' in x[1] or "RB" in x[1]:
        return True
    else:
        return False


def get_features(sentence):
    features = defaultdict(int)
    tokens = tokenize(sentence)
    pos = nltk.pos_tag(tokens)
    new_pos = list(map(lambda x: x[0], list(filter(filter_by_pos, pos))))

    for i in range(len(new_pos)):
        features[new_pos[i]] += 1
        if i < (len(new_pos) - 1):
            bigram = "%s__%s" % (new_pos[i], new_pos[i+1])
            features[bigram] += 1

    return features


def read(address):
    arq = open(address, 'rb')
    return pickle.load(arq)


def save_pickle(address, element):
    arq = open(address, 'wb')
    pickle.dump(element, arq)


def save_data_frame(s, name):
    s.to_csv(name + '.csv', index=False)


def using_queue_get_feat(d, output):
    output.put(get_features(d))


def get_all_features(data, str_domain):
    dicts = []

    file = open("DataSet/%s/features.txt" % str_domain, 'w')
    for d in data:
        feat = get_features(d)
        for f in feat.items():
            for i in range(f[1]):
                file.write(f[0])
        file.write('\n')
        dicts.append(feat)
    file.close()


def tfidf(matrix, n):
    df = defaultdict(int)
    col = matrix.columns
    for c in col:
        m = matrix[c]
        for v in m:
            if v > 0:
                df[c] += 1
    idf = {}
    for val in df.items():
        idf[val[0]] = math.log2(n / val[1])

    for c in col:
        matrix[c] *= idf[c]

    return matrix
