import numpy as np


def distance_in_adjusted_metric(X, Y, metric):
    if X.shape.__len__() == 1:
        X = X[np.newaxis, :]
    if Y.shape.__len__() == 1:
        Y = Y[np.newaxis, :]
    if metric == 'cosine':
        return cosine_distance(X, Y)
    elif metric == 'euclidean':
        return euclidean_distance(X, Y)
    else:
        raise Exception


def euclidean_distance(X, Y):
    print('eu')
    x_norms_in_sq = (X**2).sum(axis=1)
    y_norms_in_sq = (Y**2).sum(axis=1)
    scalar_product = np.dot(X, np.transpose(Y))
    result = (-2) * scalar_product
    result = result + y_norms_in_sq
    result = result + x_norms_in_sq[:, np.newaxis]
    return np.sqrt(result)


def cosine_distance(X, Y):
    scalar_product = np.dot(X, np.transpose(Y))
    x_norms = np.sqrt(np.diag(np.dot(X, np.transpose(X))))
    y_norms = np.sqrt(np.diag(np.dot(Y, np.transpose(Y))))
    normed_by_x = scalar_product / np.reshape(x_norms, (x_norms.shape[0], 1))
    normed_by_x_and_y = normed_by_x / y_norms
    return 1 - normed_by_x_and_y
