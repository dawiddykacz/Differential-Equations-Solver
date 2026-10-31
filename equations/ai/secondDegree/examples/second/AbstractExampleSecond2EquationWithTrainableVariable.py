from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *
from objects.functions.AISolution import AISolution


class AbstractExampleSecond2EquationWithTrainableVariable(AbstractExampleSecond2Equation):
    def __init__(self, solution: AISolution):
        super().__init__(solution)


class LossWithTrainableVariable(Loss):
    def __init__(self, t: TrainableVariables, with_noise: bool):
        self.__t = t

        if with_noise:
            w = [-0.082, 0.016, 0.048, -0.066, 0.042, 0.024, 0.052, 0.067, -0.056, -0.082, 0.089, 0.097, 0.082, 0.020,
                 0.086, 0.095, -0.042, 0.043, -0.048, -0.030, -0.063, -0.064, -0.043, -0.025, -0.031]
        else:
            w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0]
        self._w = tensorflow.constant(w, dtype=tensorflow.float64)
        self._w = tensorflow.reshape(self._w, [-1, 1])

        factors = [-0.9, -0.5, 0.0, 0.5, 0.9]
        grid_x, grid_y = numpy.meshgrid(factors, factors, indexing='ij')

        self._bc_x = tensorflow.constant(grid_x.reshape(-1, 1), dtype=tensorflow.float64)
        self._bc_y = tensorflow.constant(grid_y.reshape(-1, 1), dtype=tensorflow.float64)

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) * self.__t.get_variables()[0]

    @tensorflow.function
    def _condition_data(self, function, *x):
        predictions = function(self._bc_x, self._bc_y)

        exact = exact_solution(self._bc_x, self._bc_y)

        errors = tensorflow.square(predictions - exact + self._w)

        return tensorflow.reduce_sum(errors)
