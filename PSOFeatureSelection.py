import numpy as np
from SwarmPackagePy.intelligence import sw
from sklearn.metrics import accuracy_score


class PsoFeatureSelection(sw):

    def __init__(self, n, model, dimension=None, w=0.5, c1=1, c2=1, iteration=5):
        super(PsoFeatureSelection, self).__init__()
        self.__agents = None
        self.n = n
        self.dimension = dimension
        self.function = fitness_function
        self.iteration = iteration
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.model = model

    def fit(self, data, data_labels):

        self.__agents = np.random.uniform(0.0, 1.0, (self.n, self.dimension))

        tam = int(len(data)*0.8)
        train = data.iloc[:tam]
        train_labels = data_labels.iloc[:tam]
        test = data.iloc[tam:]
        test_labels = data_labels.iloc[tam:]

        velocity = np.zeros((self.n, self.dimension))
        self._points(self.__agents)
        accs = np.array([self.function(x, train, train_labels, test, test_labels, self.model)
                         for x in self.__agents])
        index = accs.argmax()

        a_best = accs[index]
        p_best = self.__agents[index]
        g_best = p_best

        historic = [g_best, [], []]
        cont = 1
        for t in range(self.iteration):

            r1 = np.random.random((self.n, self.dimension))
            r2 = np.random.random((self.n, self.dimension))
            velocity = self.w * velocity + self.c1 * r1 * (
                p_best - self.__agents) + self.c2 * r2 * (
                g_best - self.__agents)
            self.__agents += velocity
            self.__agents = np.clip(self.__agents, 0.0, 1.0)
            self._points(self.__agents)

            accs = np.array([self.function(x, train, train_labels, test, test_labels, self.model)
                             for x in self.__agents])
            index = accs.argmax()
            p_best = self.__agents[index]

            if accs[index] > a_best:
                g_best = p_best
                a_best = accs[index]

            historic[cont] = g_best
            cont += 1
            if np.array_equal(historic[0], historic[1]) and \
                    np.array_equal(historic[1], historic[2]):
                break
            if cont == 2:
                cont = 0

        self._set_Gbest(g_best)

    def transform(self, data):
        best = self.get_Gbest()

        columns = get_columns(data, best)
        new_data = data[columns]

        return new_data

    def fit_transform(self, data, labels):
        self.fit(data, labels)

        return self.transform(data)

    def get_support(self):
        return discretize(self.get_Gbest(), 'bool')


def get_columns(train, features):
    c = train.columns
    columns = []
    for i in range(len(features)):
        if features[i] == 1:
            columns.append(c[i])

    return columns


def discretize(particle, t):
    if t is 'bool':
        features = list(map(lambda x: True if x > 0.5 else False, particle))
    else:
        features = list(map(lambda x: True if x > 0.5 else False, particle))

    return features


def fitness_function(particle, train, train_labels, test, test_labels, model):
    features = discretize(particle, 'int')
    columns = get_columns(train, features)
    train = train[columns]
    test = test[columns]
    model.fit(train, train_labels)
    predicts = model.predict(test)
    acc = accuracy_score(test_labels, predicts)
    fitness = 0.95 * acc + 0.05 * (1 - (np.sum(features) / len(features)))
    return fitness

