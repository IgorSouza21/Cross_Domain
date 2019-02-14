from Spectral.SpectralGridSearch import GridSearchSpectral
import preprocess as pp


if __name__ == '__main__':
    parameters = {'nclusters': [50, 75, 100],
                  'nDI': [60, 80, 100],
                  'coocTh': [5, 10, 15],
                  'sourceFreqTh': [5, 10, 15],
                  'targetFreqTh': [5, 10, 15],
                  'gamma': [0.1, 0.5, 1.0],
                  'source': ['books', 'dvd', 'electronics', 'kitchen'],
                  'target': ['books', 'dvd', 'electronics', 'kitchen'],
                  'model': ['nb']}

    grid = GridSearchSpectral(parameters)
    grid.search()
    pp.save_pickle('Spectral/GridSearch_NB.pk', grid)
    print(grid.best)
    print(grid.best_acc)
    print('====================FINISHED=====================')
