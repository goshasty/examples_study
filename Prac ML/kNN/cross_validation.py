
import numpy as np
import nearest_neighbors


numberOfObjectsInX = 60000
foldsDefault = 3
possibleMetricScores = {
    'accuracy'
}


def concatenate_all_folds_in_list(list_folds):
    first_m = list_folds.pop(0)
    result = first_m
    for fold in list_folds:
        result = np.concatenate((result, fold))
    list_folds.insert(0, first_m)

    return result


def kfold(n, n_folds):
    list_folds = list()
    for i_fold in range(n_folds):
        ind_of_vectors_in_i_fold = \
            [el for el in range(i_fold, n, n_folds)]
        list_folds.append(np.array(ind_of_vectors_in_i_fold))

    result_list = list()
    for i in range(n_folds):
        validation_numbers = list_folds.pop(i)
        pair1 = validation_numbers
        pair0 = concatenate_all_folds_in_list(list_folds)
        list_folds.insert(i, validation_numbers)
        result_list.append((pair0, pair1))
    return result_list


def accuracy(predictions, real):
    distinguish = real - predictions
    len_test = len(distinguish)
    return len(distinguish[distinguish == 0]) / len_test


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    # algorithm = nearest_neighbors.KNNClassifier(
    #   k=k_list[-1], strategy=kwargs['strategy'], metric=kwargs['metric'],
    #   weights=kwargs['weights'], test_block_size=kwargs['test_block_size'])

    if 'alg' in kwargs.keys():
        algorithm = kwargs['alg']
    else:
        algorithm = nearest_neighbors.KNNClassifier(k_list[-1], **kwargs)

    if not cv:
        cv = kfold(X.shape[0], foldsDefault)

    result = dict()
    for _k in k_list:
        result[_k] = np.zeros((len(cv),))
    if score not in 'accuracy':
        raise TypeError
    elif score == 'accuracy':
        metric_score = accuracy

    for num_cv, fragment in enumerate(cv):
        training_set_ind = fragment[0]
        test_set_ind = fragment[1]
        try:
            algorithm.fit(X[training_set_ind], y[training_set_ind])
        except IndexError:
            print('Shape of given y is not equal X.shape[0]')
            exit(-1)
        algorithm.find_and_remember_kneighbors(X[test_set_ind])
        for another_k in k_list:
            algorithm.set_k(another_k)
            res = algorithm.predict(remembered=True, X=X[test_set_ind])
            result[another_k][num_cv] = metric_score(res, y[test_set_ind])
            # print(result[another_k][num_cv])
    return result

