from Spectral.SpectralGridSearch import run_grid
import preprocess as pp
import random
import sys

# models for run ['lr', 'knn3', 'svm_linear', 'dt', 'rf', 'nb', 'nn', 'wknn3']


if __name__ == "__main__":
    domains = ['books', 'dvd', 'electronics', 'kitchen']
    if len(sys.argv) == 4:
        model = sys.argv[1]
        eval_type = sys.argv[2]
        nfolds = int(sys.argv[3])
    elif len(sys.argv) == 3:
        model = sys.argv[1]
        eval_type = sys.argv[2]
        nfolds = 5
    elif len(sys.argv) == 2:
        model = sys.argv[1]
        eval_type = 'tt'
        nfolds = 5
    else:
        raise AttributeError('insert the parameters.')

    # grid = pp.read("C:/Users/igor_/Desktop/results/results_lr_nn_knn3_wknn3/GridSearch-lr.rs")
    # for a in grid.all_results.items():
    #     print(a)
    # print(len(grid.all_results))
    # print(grid.best_acc)
    # print(grid.best)
    #
    # model = 'lr'
    # eval_type = 'tt'
    # nfolds = 5

    file = open('seeds.txt', 'w')
    file.write(str(int(random.random() * 100)))
    file.close()

    # nclusters = 100
    # nDI = 500
    # gamma = 0.6

    for source in domains:
        for target in domains:
            if source != target:
                print('Source: %s - Target: %s' % (source, target))
                run_grid(model, source, target, eval_type, nfolds)
