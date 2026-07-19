import math

from objects.functions.Function import Function
import tensorflow
import numpy


class LossFunction(Function):
    def calculate(self, function, *x):
        y = (self._left_side_of_the_equation(function, *x) - self._right_side_of_the_equation(function, *x)
             + self._condition_in_loss(function, *x))
        conditions = self._condition(function, *x)
        conditions_data = self._condition_data(function, *x)

        y = tensorflow.reduce_mean(y ** 2)
        loss = y + self._add_condition()
        if conditions is not 0:
            conditions = tensorflow.reduce_mean(conditions ** 2)
            loss += abs(self._condition_weight()) * conditions

        if conditions_data is not 0:
            conditions_data = tensorflow.reduce_mean(conditions_data ** 2)
            loss += abs(self._condition_data_weight()) * conditions_data

        return {'loss': loss, 'loss_pde': y, 'conditions': conditions, 'conditions_data': conditions_data}

    def _left_side_of_the_equation(self, function, *x):
        return 0

    def _right_side_of_the_equation(self, function, *x):
        return 0

    def _condition_in_loss(self, function, *x):
        return 0

    def _condition(self, function, *x):
        return 0

    def _condition_data(self, function, *x):
        return 0

    def _add_condition(self):
        return 0

    def _condition_weight(self):
        return 1

    def _condition_data_weight(self):
        return 1

    def recalculate_weights(self, grads_dict, loss_error):
        return

    @staticmethod
    def max_abs_grads(grad):
        valid_grads = [g for g in grad if g is not None]

        if not valid_grads:
            return tensorflow.constant(0.0, dtype=tensorflow.float64)

        max_values = [tensorflow.reduce_max(tensorflow.abs(g)) for g in valid_grads]
        return tensorflow.reduce_max(tensorflow.stack(max_values))

    @staticmethod
    def mean_abs_grads(grad):
        valid_grads = [g for g in grad if g is not None]

        if not valid_grads:
            return tensorflow.constant(0.0, dtype=tensorflow.float64)

        max_values = [tensorflow.reduce_mean(tensorflow.abs(g)) for g in valid_grads]
        return tensorflow.reduce_mean(tensorflow.stack(max_values))
