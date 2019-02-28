import preprocess as pp
from preprocess import pd
from sklearn.model_selection import train_test_split

domains_names = ['books', 'electronics', 'dvd', 'kitchen']


def return_domain(string, train_size=0.8, test_size=0.2):
    pos, neg = pos_neg('DataSet/' + string)
    pos = pos[:5]
    neg = neg[:5]

    labels = [1] * len(pos) + [0] * len(neg)

    data = pos + neg
    domain = pp.pd.DataFrame(data, columns=['text'])

    train, test, lb_train, lb_test = train_test_split(domain, labels, train_size=train_size,
                                                      test_size=test_size, shuffle=True,
                                                      stratify=labels)
    return train['text'], test['text'], lb_train, lb_test


def dataset(nome):
    x = pp.read(nome)
    return x[0], x[1]


def get_domain(data, str_domain):
    domain = []
    lb_domain = []
    dt = list(data.values)
    for d in dt:
        if str_domain == d[0]:
            domain.append(d[2])
            lb_domain.append(d[3])
    dt = pd.DataFrame(domain, columns=['text'])
    dt['class_label'] = lb_domain

    return dt


def pos_neg(domain):
    op, lb = dataset(domain + '.pk')
    pos = []
    neg = []
    for i in range(0, len(op)-1):
        if lb[i] == 1:
            pos.append(op[i])
        else:
            neg.append(op[i])

    return pos, neg
