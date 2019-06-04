import numpy as np
import optimization


class MulticlassStrategy:
    def __init__(self, mode,  classifier=optimization.SGDClassifier,
                 num_classes=3, **kwargs):
        """
        Инициализация мультиклассового классификатора

        classifier - базовый бинарный классификатор - объект optimization

        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'

        **kwargs - параметры классификатор
        """

        if mode == 'one_vs_all':
            self.num_classes = num_classes
            self.num_classifiers = num_classes
            self.list_classifiers = list()
            for i in range(num_classes):
                self.list_classifiers.append(classifier(**kwargs))
            self.mode = 'OvA'

        elif mode == 'all_vs_all':
            self.num_classifiers = int(num_classes*(num_classes-1)/2)
            self.num_classes = num_classes
            self.list_classifiers = list()
            for i in range(0, num_classes-1):
                self.list_classifiers.append(list())
                for j in range(i+1, num_classes):
                    self.list_classifiers[i].append(classifier(**kwargs))

            self.mode = 'OvO'
        else:
            raise TypeError

    def fit(self, X, y):
        """
        Обучение классификатора
        """
        if self.mode == 'OvA':
            for num_classifier, classifier in enumerate(self.list_classifiers):
                y_this_competition = np.array(y)
                class_competitor = y == num_classifier
                y_this_competition[class_competitor] = 1
                y_this_competition[~class_competitor] = -1
                classifier.fit(X, y_this_competition, trace=True)

        elif self.mode == 'OvO':
            for i, row_classifiers in enumerate(self.list_classifiers):
                for j, classifier in enumerate(row_classifiers):
                    y_min = np.array((np.where(y == i))[0])
                    y_max = np.array((np.where(y == j+i+1))[0])
                    positions_compete = np.sort(np.concatenate((y_min, y_max)))
                    y_this_competition = np.take(y, positions_compete)
                    y_this_competition[y_this_competition == i] = -1
                    y_this_competition[y_this_competition == j] = 1
                    classifier.fit(X[positions_compete], y_this_competition,
                                   trace=True)



    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        if self.mode == 'OvA':
            margins_for_every_class = np.zeros((self.num_classes, X.shape[0]))
            # matrix shape = (K, L), then for every object (every column) find
            # max value in rows, num of row is predicted class
            for num_class, classifier in enumerate(self.list_classifiers):
                margins_for_every_class[num_class] += classifier.calc_margin(X)

            return np.argmax(
                margins_for_every_class, axis=0).reshape((X.shape[0], ))
        elif self.mode == 'OvO':
            votes_for_every_obj = np.zeros((self.num_classifiers, X.shape[0]))
            num_row = 0
            for i, row_classifiers in enumerate(self.list_classifiers):
                for j, classifier in enumerate(row_classifiers):
                    predicts = classifier.predict(X)
                    predicts[predicts == 1] = i+j+1
                    predicts[predicts == -1] = i
                    votes_for_every_obj[num_row] += predicts
                    num_row += 1

            def get_most_frequent_class(array_classes):
                array_classes = array_classes.ravel().astype(int)
                return np.argmax(np.bincount(array_classes))

            #return np.apply_along_axis(get_most_frequent_class, 0,
                                       votes_for_every_obj.astype(int))