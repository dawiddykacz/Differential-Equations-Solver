from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *
from objects.functions.AISolution import AISolution


class AbstractExampleFirst2ProblemWithTrainableVariable(AbstractExampleFirst2Problem):
    def __init__(self, solution: AISolution):
        super().__init__(solution)


class LossWithTrainableVariable(Loss):
    def __init__(self, t: TrainableVariables, with_noise: bool):
        self.__t = t

        if with_noise:
            w = [-0.082, 0.016, 0.048, -0.066, 0.042]
        else:
            w = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.__w = tensorflow.constant(w, dtype=tensorflow.float64)

        self.__points = tensorflow.constant([-1.0, -0.5, 1.0, 0.5, 0.1], dtype=tensorflow.float64)

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) * self.__t.get_variables()[0]

    def _condition_data(self, function, *x):
        base_x = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        results = []
        for i in range(5):
            point_val = self.__points[i] * base_x
            noise_val = self.__w[i]

            results.append(condition_bc(function, point_val, noise_val))

        return tensorflow.add_n(results)
