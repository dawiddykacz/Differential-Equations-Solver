from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService
from solvers.models.ChooseModel import WangParams
from solvers.models.ModelConfiguration import ModelConfiguration, ModelWithOptimizationConfiguration
from tasks.ai.article.examples.ArticleExamplesImport import *
from solvers.AISolver import set_learning_rate
from services.TaskService import set_equations_amount

from TestBuilder import test_1
from TestBuilder import test_2

model_configuration = None


def configure_solver():
    model_with_optimization_configuration = ModelWithOptimizationConfiguration(epochs=5000)
    model_configuration = ModelConfiguration()
    wang_params = WangParams(hidden_dim=50, activation_function='tanh')
    model_configuration.configure(model_with_optimization=None,
                                  wang_configuration=None,
                                  dense_list=[
                                      tensorflow.keras.layers.Dense(units=50, activation='tanh', dtype='float64'),
                                      tensorflow.keras.layers.Dense(units=50, activation='tanh', dtype='float64'),
                                      tensorflow.keras.layers.Dense(units=50, activation='tanh', dtype='float64'),
                                  ])


def run_all(learning_rate: float):
    set_learning_rate(learning_rate)
    set_equations_amount(2)
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    test_1(task_repository)

    task_service.solve(10)
    weight_plot_service.plots(task_service.get_task_dict(), task_service.get_epochs())
    error_messages = task_service.get_error_messages()
    if error_messages is not None:
        for error_message in error_messages:
            print(error_message)


if __name__ == '__main__':
    configure_solver()
    run_all(10 ** -3)
