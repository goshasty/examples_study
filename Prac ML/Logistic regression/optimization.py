import numpy as np
import oracles
import frequently_functions
import time

class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.max_iter = max_iter
        self.alpha = step_alpha
        self.beta = step_beta
        self.tolerance = tolerance
        self.w = None
        if loss_function == 'binary_logistic':
            self.obj = oracles.BinaryLogistic(**kwargs)
        elif loss_function == 'multinomial_logistic':
            self.obj = oracles.MulticlassLogistic(**kwargs)

        self.gr_loss_func = self.obj.grad
        self.loss_func = self.obj.func

    def one_iteration(self, X, y, num_iter):
        start_time = time.time()
        eta = self.alpha/np.power(num_iter, self.beta)
        self.w = self.w - eta*self.get_gradient(X, y)
        finish_time = time.time()
        time_one_iter = finish_time-start_time
        return time_one_iter

    @staticmethod
    def get_accuracy_test(predict, y_test):
        return ((predict - y_test) == 0).astype(int).sum() / y_test.shape[0]

    def fit(self, X, y, w_0=None, trace=False,
            accuracy=False, x_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if type(w_0) != np.ndarray:
            if type(self.obj) is oracles.BinaryLogistic:
                self.w = np.zeros(X.shape[1])
            elif type(self.obj) is oracles.MulticlassLogistic:
                self.w = np.zeros((np.max(y)+1, X.shape[1]))
        else:
            self.w = w_0

        num_iter = 0

        history = dict()
        history['time'] = list()
        history['func'] = list()
        if accuracy:
            history['accuracy'] = list()

        prev_func_loss = self.get_objective(X, y)
        history['func'].append(prev_func_loss)
        while num_iter < self.max_iter:

            num_iter += 1
            time_one_iter = self.one_iteration(X, y, num_iter)
            history['time'].append(time_one_iter)
            now_func_loss = self.get_objective(X, y)
            history['func'].append(now_func_loss)
            if accuracy:
                accuracy_current = frequently_functions.get_accuracy_test(
                    self.predict(x_test), y_test)
                history['accuracy'].append(accuracy_current)
                # print(accuracy_current)
            else:
                accuracy_current = None
            if np.abs(prev_func_loss - now_func_loss) < self.tolerance:
                break
            prev_func_loss = now_func_loss
            print(now_func_loss)
            # print(num_iter, '%.4f  %.5f %.5f' %
            #       (time_one_iter, prev_func_loss, accuracy_current))
        print(num_iter)
        if trace:
            return history


    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        if isinstance(self.obj, oracles.BinaryLogistic):
            if oracles.BaseSmoothOracle.is_sparse(X):
                return np.sign(X*self.w.T).astype(int)
            else:
                return np.sign(
                    np.dot(X, self.w.T)).reshape(X.shape[0], ).astype(int)
        else:
            if oracles.BaseSmoothOracle.is_sparse(X):
                return np.argmax((X * self.w.T), axis=1)
            else:
                return np.argmax(np.dot(X, self.w.T), axis=1)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        pass

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.loss_func(X, y, self.w)

    def calc_margin(self, X):
        """
        :param X: Set of objects shape=(L, D)
        :return: margin for every object regarding 2 classes shape=(L, )
        """
        return X.dot(self.w).reshape((X.shape[0], ))

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.gr_loss_func(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic', batch_size=1000, step_alpha=1.5, step_beta=0.3,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход


        max_iter - максимальное число итераций

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """

        super().__init__(loss_function, step_alpha, step_beta,
                         tolerance, max_iter, **kwargs)
        self.batch_size = batch_size
        self.random_seed = random_seed
        print(self.batch_size, self.alpha, self.beta)

    def batch_ind_generator(self, permutation, num_obj):
        cur_pos = 0
        while cur_pos < num_obj:
            cur_indexes = permutation[cur_pos:cur_pos+self.batch_size]
            yield cur_indexes
            cur_pos += self.batch_size

    def fit(self, X, y, w_0=None, trace=False, log_freq=1,
            accuracy=False, x_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности
        векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if type(w_0) != np.ndarray:
            if type(self.obj) is oracles.BinaryLogistic:
                self.w = np.zeros(X.shape[1])
            elif type(self.obj) is oracles.MulticlassLogistic:
                self.w = np.zeros((np.max(y)+1, X.shape[1]))
        else:
            self.w = w_0

        approx_age_cur = 0
        total_ages = 0
        num_obj = X.shape[0]

        history = dict()
        history['epoch_num'] = list()
        history['time'] = list()
        history['func'] = list()
        history['weights_diff'] = list()
        if accuracy:
            history['accuracy'] = list()
            accuracy_current = 0
        prev_func_loss = 0
        prev_w = self.w
        start_time = time.time()
        while total_ages < self.max_iter:
            permutation = np.random.permutation(num_obj)

            for batch_ind in self.batch_ind_generator(permutation, num_obj):
                approx_age_cur += self.batch_size/num_obj
                total_ages += self.batch_size/num_obj

                eta = self.alpha / np.power(round(total_ages)+1, self.beta)
                self.w = self.w - eta * self.get_gradient(
                    X[batch_ind], y[batch_ind])

                if approx_age_cur >= log_freq:
                    approx_age_cur = 0
                    history['epoch_num'].append(total_ages)
                    history['time'].append(time.time() - start_time)
                    history['func'].append(self.get_objective(X, y))
                    history['weights_diff'].append(
                        (np.linalg.norm(self.w-prev_w))**2)

                    if accuracy:
                        accuracy_current = GDClassifier.get_accuracy_test(
                            self.predict(x_test), y_test)
                        history['accuracy'].append(accuracy_current)

                    prev_w = self.w
                    start_time = time.time()
            now_func_loss = self.get_objective(X, y)

            if abs(prev_func_loss - now_func_loss) < self.tolerance:
                break
            prev_func_loss = now_func_loss
            # print(now_func_loss)
        if trace:
            print(np.array(history['time']).sum())
            print('----')
            return history
