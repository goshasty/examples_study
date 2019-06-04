import numpy as np
import scipy.special
import scipy.sparse


class BaseSmoothOracle:
    """
    Base class for oracles
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    @staticmethod
    def is_sparse(x):
        return isinstance(x, scipy.sparse.csr_matrix)

class BinaryLogistic(BaseSmoothOracle):
    """
    Oracle for binary logistic regression

    L2 regularization is available
    """

    def __init__(self, l2_coef=0):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        w.reshape(-1, 1)
        num_objs = X.shape[0]
        if BaseSmoothOracle.is_sparse(X):
            inner_prod = X*w
        else:
            inner_prod = np.dot(X, w)

        all_loss = np.logaddexp(np.zeros((X.shape[0],)), -y*inner_prod)

        return all_loss.sum()/num_objs + self.l2_coef/2*np.linalg.norm(w)**2

        # return super().func(w)

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        w.reshape(-1, 1)
        if BaseSmoothOracle.is_sparse(X):
            inner_prod = X*w
        else:
            inner_prod = np.dot(X, w)
        margin = y*inner_prod
        derivation_external = -y * scipy.special.expit(-margin)
        l2_reg = self.l2_coef*w
        
        if BaseSmoothOracle.is_sparse(X):
            derivation_external_mul_X_summed = (
                derivation_external.reshape(1, -1)*X)
        else:
            derivation_external_mul_X_summed = np.dot(
                derivation_external.reshape(1, -1), X)

        return (derivation_external_mul_X_summed.reshape(w.shape) / X.shape[0]
                + l2_reg)



class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.

    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """

    def __init__(self, class_number=0, l2_coef=0):
        """
        Задание параметров оракула.

        class_number - количество классов в задаче

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef
        self.class_n = class_number



    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - двумерный numpy array
        """
        if self.class_n == 0:
            self.class_n = y.shape[0]
        if BaseSmoothOracle.is_sparse(X):
            denominator = np.sum(scipy.special.logsumexp(X*w.T, axis=1))
            prob_true_class = X.multiply(w[y, :]).sum()
        else:
            denominator = np.sum(scipy.special.logsumexp(np.dot(X, w.T), axis=1))
            prob_true_class = (X*w[y, :]).sum()

        norms = self.l2_coef/2*(np.power(np.linalg.norm(w, axis=1), 2)).sum()
        return (-prob_true_class + denominator)/X.shape[0] + norms


        # return super().func(w)

    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - двумерный numpy array
        """

        inner_product = X.dot(w.T)
        max_inner_prod_for_every_obj = np.max(inner_product, axis=1).reshape(-1, 1)
        inner_product_trick = inner_product - max_inner_prod_for_every_obj
        norm_for_every_obj = np.sum(np.exp(inner_product_trick), axis=1)
        norm_for_every_obj = norm_for_every_obj.reshape(-1, 1)
        exp_and_normed_inner = np.exp(inner_product_trick) / norm_for_every_obj
        grid_of_true_classes = np.zeros((w.shape[0], X.shape[0]))
        grid_of_true_classes[y, range(X.shape[0])] = -1

        external_derivation = grid_of_true_classes + exp_and_normed_inner.T
        if BaseSmoothOracle.is_sparse(X):
            external_derivation_summed = external_derivation * X
        else:
            external_derivation_summed = external_derivation.dot(X)

        total = external_derivation_summed/X.shape[0]
        norms = self.l2_coef * w
        return total + norms

