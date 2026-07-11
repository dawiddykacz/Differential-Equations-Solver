from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService
from solvers.models.ModelConfiguration import ModelConfiguration, ModelWithOptimizationConfiguration

from tasks.ai.article.examples.ArticleExamplesImport import *
from tasks.ai.secondDegree.SecondDegreeImport import *
from solvers.AISolver import set_learning_rate
from services.TaskService import set_equations_amount

model_configuration = None


def configure_solver():
    model_with_optimization_configuration = ModelWithOptimizationConfiguration(epochs=500)
    model_configuration = ModelConfiguration()
    model_configuration.configure(model_with_optimization= model_with_optimization_configuration)


def run_all(learning_rate: float):
    set_learning_rate(learning_rate)
    set_equations_amount(1)

    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    task_repository.add_task(FirstProblemLossTask())
    task_repository.add_task(FirstProblemLossWithWeightTask())
    # task_repository.add_task(FirstProblemLossTask())
    # task_repository.add_task(Second2ProblemLossWithNoiseTask())
    # task_repository.add_task(FifthProblemSimpleTask())
    # task_repository.add_task(FifthProblemLossTask())
    # task_repository.add_task(Second2ProblemLossTask())
    # task_repository.add_task(Second2ProblemLossWithNoiseTask())
    # task_repository.add_task(Fourth2EquationLossTask())
    task_service.solve(4000)
    weight_plot_service.plots(task_service.get_task_dict(), task_service.get_epochs())

    error_messages = task_service.get_error_messages()
    if error_messages is not None:
        for error_message in error_messages:
            print(error_message)


if __name__ == '__main__':
    configure_solver()
    run_all(0.1)
    # run_all(0.093201)
