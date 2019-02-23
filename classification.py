from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from NeuralNetwork import NeuralNetwork
from timeit import default_timer

model_names = ['knn3', 'knn5', 'knn7', 'wknn3', 'wknn5', 'wknn7', 'nb',
               'dt', 'svm_linear', 'lr', 'nn']


def choose_model(str_model, k_features=None):
    if str_model == 'knn3':
        model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=4)
    elif str_model == 'knn5':
        model = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=4)
    elif str_model == 'knn7':
        model = KNeighborsClassifier(n_neighbors=7, metric='euclidean', n_jobs=4)
    elif str_model == 'wknn3':
        model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean', n_jobs=4)
    elif str_model == 'wknn5':
        model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean', n_jobs=4)
    elif str_model == 'wknn7':
        model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean', n_jobs=4)
    elif str_model == 'nb':
        model = GaussianNB()
    elif str_model == 'dt':
        model = DecisionTreeClassifier()
    elif str_model == 'rf':
        model = RandomForestClassifier()
    elif str_model == 'svm_linear':
        model = svm.LinearSVC()
    elif str_model == 'lr':
        model = LogisticRegression(solver='lbfgs', n_jobs=4)
    elif str_model == 'nn':
        model = NeuralNetwork(k_features, n_hidden_layers=3,
                              n_neurons=[int(k_features/2), int(k_features/5), 20])
    else:
        model = None

    return model


def run_classifier(tuple_classifier, train_dt, train_lb, test_dt, test_lb):
    start = default_timer()
    tuple_classifier[0].fit(train_dt, train_lb)
    predict_label = tuple_classifier[0].predict(test_dt)
    end = default_timer()
    tuple_classifier[1].append(accuracy_score(test_lb, predict_label))
    tuple_classifier[2].append(end - start)


def train_and_test(train_dt, train_lb, test_dt, test_lb, models, k_features=None):
    classifiers = [[choose_model(a, k_features), [], []] for a in model_names if a in models]

    for classifier in classifiers:
        run_classifier(classifier, train_dt, train_lb, test_dt, test_lb)

    return [(a[1], a[2]) for a in classifiers]


def train_test_one_model(train_dt, train_lb, test_dt, test_lb, model, k_features=None):
    classifier = [choose_model(model, k_features), [], []]
    run_classifier(classifier, train_dt, train_lb, test_dt, test_lb)

    print('finish classification')
    return classifier[1][0], classifier[2][0]
