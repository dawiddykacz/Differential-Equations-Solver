import tensorflow
import numpy
from tqdm import tqdm

from objects.space.Space import Space
from objects.functions.loss.LossFunction import LossFunction
from objects.TrainableVariables import TrainableVariables


class AISolver:
    def __init__(self, space: Space, solution_function, loss_function: LossFunction,
                 trainable_variables: TrainableVariables
                 = TrainableVariables(), plots: bool = True):
        self.__neural_network = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Dense(units=10, activation='sigmoid', dtype='float64'),
            tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')])
        self.__optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
        self.__points = space.get_points_to_neural_network()
        self.__solution_function = solution_function
        self.__loss_function = loss_function
        self.__plots = plots

        if trainable_variables is None:
            self.__trainable_variables = TrainableVariables()
        else:
            self.__trainable_variables = trainable_variables

        self.__trainable_plot = []

        for i in self.__trainable_variables.get_variables():
            self.__trainable_plot.append([])

        self.__loss_array = numpy.array([])

        if len(self.__points) > 1:
            self.__inputs = tensorflow.concat(self.__points, axis=1)
        else:
            self.__inputs = self.__points[0]

    def calculate(self, *variables):
        return self.__neural_network(tensorflow.concat(variables, axis=1))

    def solve(self, epochs: int):
        self.__neural_network(self.__inputs)

        for i in tqdm(range(epochs)):
            with tensorflow.GradientTape() as tape:
                current_loss = self.__loss_function.calculate(self.__solution_function, *self.__points)
                if self.__plots:
                    self.__loss_array = numpy.append(self.__loss_array, current_loss.numpy())

                    for i in range(len(self.__trainable_plot)):
                        self.__trainable_plot[i].append(self.__trainable_variables.get_variables()[i].numpy())
            grads = tape.gradient(current_loss, self.__neural_network.trainable_variables +
                                  self.__trainable_variables.get_variables())
            self.__optimizer.apply_gradients(zip(grads, self.__neural_network.trainable_variables +
                                                 self.__trainable_variables.get_variables()))

    def get_loss_array(self):
        if self.__plots:
            return self.__loss_array
        return None

    def get_trainable_variables_array(self):
        if self.__plots:
            return self.__trainable_plot
        return None
