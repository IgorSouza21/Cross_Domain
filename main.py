from Spectral.SpectralGridSearch import GridSearchSpectral
import preprocess as pp
import sys

# models for run ['lr', 'knn3', 'svm_linear', 'dt', 'rf', 'nb', 'nn', 'wknn3']


def run_grid(model, processes):
    parameters = {'nclusters': [2, 5, 7],
                  'nDI': [60, 80, 100],
                  'coocTh': [5, 10, 15],
                  'sourceFreqTh': [5, 10, 15],
                  'targetFreqTh': [5, 10, 15],
                  'gamma': [0.1, 0.5, 1.0],
                  'source': ['books', 'dvd', 'electronics', 'kitchen'],
                  'target': ['books', 'dvd', 'electronics', 'kitchen'],
                  'model': model}

    grid = GridSearchSpectral(parameters)
    grid.search(processes)
    pp.save_pickle('Spectral/results/GridSearch-' + model[0] + '.rs', grid)
    print(grid.best)
    print(grid.best_acc)
    print('====================FINISHED=====================')


if __name__ == "__main__":
    run_grid(str(sys.argv[1]))
    # run_grid(['lr'], 2)
