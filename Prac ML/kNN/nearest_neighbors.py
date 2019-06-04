
from sklearn.neighbors import NearestNeighbors
import numpy as np
import distances

epsilon = 10e-5
myOwnStrategy = 'my_own'
numClassMin = 0

availableMetrics = {
    'cosine',
    'euclidean'
}

availableStrategies = {
    'my_own',
    'brute',
    'kd_tree',
    'ball_tree'
}


class KNNClassifier:

    def __init__(self, k, strategy, metric, weights, test_block_size):
        if metric not in availableMetrics:
            raise Exception
        if strategy == myOwnStrategy:
            self.matrix_objects_X = None
        elif strategy in availableStrategies:
            self.sklearn_KNN_algorithm = NearestNeighbors(
                n_neighbors=k, algorithm=strategy, metric=metric)
        else:
            raise Exception

        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.vector_values_y = None
        self.rem_test_distances_to_k_nn = None
        self.rem_test_ind_to_k_nn = None
        self.remember_k_nn_to_test = False
        self.num_classes = 0
        self.num_min_class = 0

    def fit(self, X, y):
        self.vector_values_y = y
        self.num_min_class = numClassMin
        self.num_classes = np.max(y) + 1 - self.num_min_class
        self.forget_remembered_k_nn()
        if self.strategy != myOwnStrategy:
            self.sklearn_KNN_algorithm.fit(X, y)
        else:
            self.matrix_objects_X = X

    def find_and_remember_kneighbors(self, X):
        print('find and remember')
        self.rem_test_distances_to_k_nn, self.rem_test_ind_to_k_nn = \
            self.find_kneighbors(X, True)
        self.remember_k_nn_to_test = True

    def forget_remembered_k_nn(self):
        self.rem_test_ind_to_k_nn = None
        self.rem_test_distances_to_k_nn = None
        self.remember_k_nn_to_test = None

    def find_knn_for_one_test_block(self, test_block, return_distance=True):
        '''

        :param test_block: block of vectors from test set
        :param return_distance: if True -> return matrix of distances to knn
                                if False -> not
        :return: distances (if return_distance) and indexes to knn of given
                 test_block (shape[0] = self.test_block_size)
        '''
        if self.strategy != myOwnStrategy:
            return self.sklearn_KNN_algorithm.kneighbors(
                test_block, self.k, return_distance)
        if test_block.shape[0] == 1:
            test_block = test_block[np.newaxis, :]

        matrix_distances = distances.distance_in_adjusted_metric(
            test_block, self.matrix_objects_X, self.metric)

        # tb = test_block
        dist_neighbours_to_tb = np.zeros(shape=(test_block.shape[0], self.k))
        indexes_neighbours_to_tb = np.zeros(
            shape=(test_block.shape[0], self.k), dtype=int)

        for i, obj in enumerate(matrix_distances):
            indexes_neighbours = np.argpartition(obj, self.k)[:, :self.k]
            indexes_neighbours = np.argsort(obj[indexes_neighbours])
            dist_neighbours_to_tb[i] = obj[indexes_neighbours]
            indexes_neighbours_to_tb[i] = indexes_neighbours

        if dist_neighbours_to_tb.shape[0] == 1:
            dist_neighbours_to_tb = dist_neighbours_to_tb[np.newaxis, :]
        if indexes_neighbours_to_tb.shape[0]() == 1:
            indexes_neighbours_to_tb = indexes_neighbours_to_tb[np.newaxis, :]

        if return_distance:
            return dist_neighbours_to_tb, indexes_neighbours_to_tb
        else:
            return indexes_neighbours_to_tb

    def find_kneighbors(self, X, return_distance):
        num_obj = X.shape[0]
        if self.test_block_size == 0 or self.test_block_size >= num_obj:
            print('no blocks')
            return self.find_knn_for_one_test_block(X, return_distance)

        quantity_blocks = num_obj // self.test_block_size
        if num_obj % self.test_block_size != 0:
            quantity_blocks += 1

        indexes_knn_all_blocks = np.zeros((X.shape[0], self.k), dtype=int)
        dist_knn_all_blocks = np.zeros((X.shape[0], self.k), dtype=np.float)

        for num_block in range(0, quantity_blocks):
            print('block number ' + str(num_block))
            first_ind = num_block * self.test_block_size
            last_ind = first_ind + self.test_block_size
            this_block_res = self.find_knn_for_one_test_block(
                X[first_ind:last_ind], return_distance)
            indexes_knn_all_blocks[first_ind:last_ind] = this_block_res[1]
            dist_knn_all_blocks[first_ind:last_ind] = this_block_res[0]

        if dist_knn_all_blocks.shape[0] == 1:
            dist_knn_all_blocks = dist_knn_all_blocks[np.newaxis, :]
        if indexes_knn_all_blocks.shape[0] == 1:
            indexes_knn_all_blocks = indexes_knn_all_blocks[np.newaxis, :]

        if return_distance:
            return dist_knn_all_blocks, indexes_knn_all_blocks
        else:
            return indexes_knn_all_blocks

    @staticmethod
    def vote_with_weights(dist):
        return 1/(dist + epsilon)

    def _predict_for_one_obj_weights(self, indexes_to_k_nn, dist_to_k_nn):
        votes_for_classes = np.zeros(shape=self.num_classes)
        '''
        we need to indexes_to_k_nn[:self.k] because
        indexes_to_k_nn could contain more nn than we need
        due find_and_remember_kneighbors
        
        '''
        for num_nn, ind_nn_in_X in enumerate(indexes_to_k_nn[:self.k]):
            votes_for_classes[
                self.vector_values_y[ind_nn_in_X] - self.num_min_class
            ] += self.vote_with_weights(dist_to_k_nn[num_nn])

        return np.argmax(votes_for_classes)

    def _predict_for_one_obj_no_w(self, indexes_to_k_nn):
        votes_for_classes = np.zeros(shape=self.num_classes)
        for ind_nn_in_X in indexes_to_k_nn[:self.k]:
            votes_for_classes[
                self.vector_values_y[ind_nn_in_X]-self.num_min_class
            ] += 1
        return np.argmax(votes_for_classes)

    def predict(self, remembered, X):
        if X.shape.__len__() == 1:
            X = X[np.newaxis, :]
        prediction = np.zeros(shape=X.shape[0], dtype=int)
        if remembered:
            if not self.remember_k_nn_to_test:
                raise Exception
            dist = self.rem_test_distances_to_k_nn
            indexes = self.rem_test_ind_to_k_nn
        else:
            dist, indexes = self.find_kneighbors(X, True)

        if self.weights:
            for i in range(0, len(X)):
                prediction[i] = self._predict_for_one_obj_weights(
                                    indexes[i], dist[i])\
                                + self.num_min_class
        else:
            for i in range(0, len(X)):
                prediction[i] = self._predict_for_one_obj_no_w(indexes[i])\
                                + self.num_min_class
        return prediction

    def set_metric(self, new_metric):
        self.forget_remembered_k_nn()
        self.metric = new_metric

    def set_k(self, new_k):
        self.k = new_k

    def set_weights(self, new_weights):
        self.weights = new_weights


def get_ind_test_obj_wrong_predict(predict, real):
    distinguish = predict - real
    a = np.nonzero(distinguish)[0]
    return a
