from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService
from solvers.models.ChooseModel import WangParams
from solvers.models.ModelConfiguration import ModelConfiguration, ModelWithOptimizationConfiguration

from tasks.ai.article.examples.ArticleExamplesImport import *
from tasks.ai.secondDegree.SecondDegreeImport import *
from solvers.AISolver import set_learning_rate
from services.TaskService import set_equations_amount

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


def basic_rep(task_repository):
    range_weight = 1 / (0.1 ** 4)

    for with_noise in [False, True]:
        for weight in [1, range_weight]:
            task_repository.add_task(ExampleFirst2ProblemLossTask(weight_data=weight, weight_conditions=weight,
                                                                  with_noise=with_noise))
            task_repository.add_task(ExampleSimpleFirst2ProblemLossTask(weight_data=weight,
                                                                        with_noise=with_noise))
            task_repository.add_task(ExampleSecond2ProblemLossTask(weight_data=weight, weight_conditions=weight,
                                                                   with_noise=with_noise))
            task_repository.add_task(ExampleSimpleSecond2ProblemLossTask(weight_data=weight,
                                                                         with_noise=with_noise))
            for alpha in [0.9]:
                task_repository.add_task(
                    ExampleFirst2ProblemLossWithWeightTask(alpha=alpha, weight_conditions=weight,
                                                           weight_data=weight, with_noise=with_noise))
                task_repository.add_task(
                    ExampleSimpleFirst2ProblemLossWithWeightTask(alpha=alpha, weight_data=weight,
                                                                 with_noise=with_noise))
                task_repository.add_task(
                    ExampleSecond2ProblemLossTaskWithWeightTask(alpha=alpha, weight_conditions=weight,
                                                                weight_data=weight, with_noise=with_noise))
                task_repository.add_task(
                    ExampleSimpleSecond2ProblemLossTaskWithWeightTask(alpha=alpha, weight_data=weight,
                                                                      with_noise=with_noise))


def run_all(learning_rate: float):
    set_learning_rate(learning_rate)
    set_equations_amount(10)

    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    d_min = 0.0001
    range_weight = 10 ** 4
    for with_noise in [False, True]:
        task_repository.add_task(ExampleFirst2ProblemLossTask(weight_data=1,
                                                              weight_pde=d_min,
                                                              weight_conditions=1,
                                                              with_noise=with_noise))
        task_repository.add_task(ExampleSimpleFirst2ProblemLossTask(weight_pde=d_min,
                                                                    weight_data=1,
                                                                    with_noise=with_noise))

        task_repository.add_task(
            ExampleFirst2ProblemLossWithWeightTask(alpha=0.9,
                                                   weight_pde=d_min,
                                                   weight_conditions=1,
                                                   weight_data=1, with_noise=with_noise))
        task_repository.add_task(
            ExampleSimpleFirst2ProblemLossWithWeightTask(alpha=0.9,
                                                         weight_pde=d_min,
                                                         weight_data=1,
                                                         with_noise=with_noise))

        for weight in [1, range_weight]:
            task_repository.add_task(ExampleFirst2ProblemLossTask(weight_data=weight,
                                                                  weight_pde=1,
                                                                  weight_conditions=weight,
                                                                  with_noise=with_noise))
            task_repository.add_task(ExampleSimpleFirst2ProblemLossTask(weight_pde=1,
                                                                        weight_data=weight,
                                                                        with_noise=with_noise))
            for alpha in [0.9]:
                task_repository.add_task(
                    ExampleFirst2ProblemLossWithWeightTask(alpha=alpha,
                                                           weight_pde=1,
                                                           weight_conditions=weight,
                                                           weight_data=weight, with_noise=with_noise))
                task_repository.add_task(
                    ExampleSimpleFirst2ProblemLossWithWeightTask(alpha=alpha,
                                                                 weight_pde=1,
                                                                 weight_data=weight,
                                                                 with_noise=with_noise))

    task_service.solve(5000)
    weight_plot_service.plots(task_service.get_task_dict(), task_service.get_epochs())

    error_messages = task_service.get_error_messages()
    if error_messages is not None:
        for error_message in error_messages:
            print(error_message)


if __name__ == '__main__':
    configure_solver()
    run_all(10 ** -3)
