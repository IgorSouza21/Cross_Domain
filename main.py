from Spectral.SpectralGridSearch import GridSearchSpectral
import preprocess as pp
from matplotlib import pyplot as pl


def print_measures(dataframe, algorithms):
    import numpy as np
    for a in algorithms:
        print(a)
        print('media -> ', np.mean(dataframe[a]))
        print('mediana -> ', np.median(dataframe[a]))
        print('desvio padrão -> ', np.std(dataframe[a]))
        print('================')


def plot_bar_x(label, accuracies, texto):
    import numpy as np
    index = np.arange(len(label))
    pl.bar(index, accuracies)
    pl.xlabel('Algoritmo', fontsize=8)
    pl.ylabel('Média', fontsize=8)
    pl.xticks(index, label, fontsize=9, rotation=30)
    pl.title(texto)
    pl.show()


# labels = ['kmeans', 'PSO', 'Hybrid']
# data = preprocess.pd.read_csv('resultados/cluster_wss.csv')
# mean = data.mean()
# median = data.median()
# std = data.std()
# print(mean)
# print(median)
# print(std)
# plot_bar_x(labels, mean, 'Clustering')

if __name__ == '__main__':
    parameters = {'nclusters': [100, 50],
                  'nDI': [60, 100],
                  'coocTh': [5, 10],
                  'sourceFreqTh': [5, 10],
                  'targetFreqTh': [5, 10],
                  'gamma': [0.5, 1.0],
                  'source': ['books', 'dvd', 'electronics', 'kitchen'],
                  'target': ['books', 'dvd', 'electronics', 'kitchen'],
                  'model': ['lr', 'knn3']}

    grid = GridSearchSpectral(parameters)
    grid.search()
    pp.save_pickle('Spectral/GridSearch', grid)
    print(grid.best)
    print(grid.best_acc)
