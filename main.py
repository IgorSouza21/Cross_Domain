from Spectral.SpectralGridSearch import GridSearchSpectral
import preprocess as pp
import sys

# models for run ['lr', 'knn3', 'svm_linear', 'dt', 'rf', 'nb', 'nn', 'wknn3']


def run_grid(model, eval=None, nfold=None):
    parameters = {'nclusters': [2, 5, 7],
                  'nDI': [60, 80, 100],
                  'coocTh': [5, 10, 15],
                  'sourceFreqTh': [5, 10, 15],
                  'targetFreqTh': [5, 10, 15],
                  'gamma': [0.1, 0.5, 1.0],
                  'source': ['books', 'dvd', 'electronics', 'kitchen'],
                  'target': ['books', 'dvd', 'electronics', 'kitchen'],
                  'model': model}

    # parameters = {'nclusters': [5, 7],
    #               'nDI': [80, 100],
    #               'coocTh': [10, 15],
    #               'sourceFreqTh': [10, 15],
    #               'targetFreqTh': [10, 15],
    #               'gamma': [0.5, 1.0],
    #               'source': ['books', 'dvd', 'electronics', 'kitchen'],
    #               'target': ['books', 'dvd', 'electronics', 'kitchen'],
    #               'model': model}

    grid = GridSearchSpectral(parameters, nfold)
    grid.search(eval)
    pp.save_pickle('Spectral/results/GridSearch-' + model[0] + '.rs', grid)
    print("Best Result Found")
    print(grid.best)
    print(grid.best_acc)
    print('====================FINISHED=====================')


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_grid([sys.argv[1]], sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) == 3:
        run_grid([sys.argv[1]], sys.argv[2])

    # grid = pp.read("C:/Users/igor_/PycharmProjects/Cross_Domain/Spectral/results/GridSearch-lr.rs")
    # print(grid.all_results)
    # run_grid(['lr'], 'kf')
