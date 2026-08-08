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
    wang_params = WangParams()
    model_configuration.configure(model_with_optimization=None,
                                  wang_configuration=None)


def run_all(learning_rate: float):
    set_learning_rate(learning_rate)
    set_equations_amount(20)

    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    for with_noise in [True, False]:
        for weight in [1, 3, 5, 7, 10, 12, 15, 17, 20]:
            task_repository.add_task(ExampleFirst2ProblemLossTask(weight=weight, with_noise=with_noise))
            task_repository.add_task(ExampleSecond2ProblemLossTask(weight=weight, with_noise=with_noise))
        for alpha in [0.1]:
            for alpha_lower in [1]:
                task_repository.add_task(
                    ExampleFirst2ProblemLossWithWeightTask(alpha=alpha, alpha_lower=alpha_lower,
                                                           with_noise=with_noise))
                task_repository.add_task(
                    ExampleSecond2ProblemLossTaskWithWeight(alpha=alpha, alpha_lower=alpha_lower,
                                                            with_noise=with_noise))

    task_service.solve(5000)
    weight_plot_service.plots(task_service.get_task_dict(), task_service.get_epochs())

    error_messages = task_service.get_error_messages()
    if error_messages is not None:
        for error_message in error_messages:
            print(error_message)


if __name__ == '__main__':
    configure_solver()
    run_all(0.1)
