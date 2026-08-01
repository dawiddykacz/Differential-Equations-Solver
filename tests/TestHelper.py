from repositories import TaskRepository
from repositories.TaskRepository import TasksRepository
from solvers.models.ModelConfiguration import ModelWithOptimizationConfiguration, ModelConfiguration, WangParams
from tasks.ai.article.examples.ArticleExamplesImport import *
from tasks.ai.secondDegree.SecondDegreeImport import *


def add_all_first_degree(task_repository: TaskRepository):
    task_repository.add_task(FirstProblemSimpleTask())
    task_repository.add_task(FirstProblemLossTask())
    task_repository.add_task(FirstProblemLossWithWeightTask())

    task_repository.add_task(SecondProblemSimpleTask())
    task_repository.add_task(SecondProblemLossTask())
    task_repository.add_task(SecondProblemLossWithWeightTask())

    task_repository.add_task(ThirdProblemSimpleTask())
    task_repository.add_task(ThirdProblemLossTask())
    task_repository.add_task(ThirdProblemLossWithWeightTask())

    task_repository.add_task(FifthProblemSimpleTask())
    task_repository.add_task(FifthProblemLossTask())
    task_repository.add_task(FifthProblemWithDistanceFunctionTask())
    task_repository.add_task(FifthProblemLossWithPointTask())
    task_repository.add_task(FifthProblemLossWithWeightTask())

    task_repository.add_task(SixthProblemSimpleTask())
    task_repository.add_task(SixthProblemLossTask())
    task_repository.add_task(SixthProblemLossWithWeightTask())

    task_repository.add_task(SeventhProblemSimpleTask())
    task_repository.add_task(SeventhProblemLossTask())
    task_repository.add_task(SeventhProblemLossWithWeightTask())


def add_all_second_degree(task_repository: TaskRepository):
    task_repository.add_task(ExampleFirst2ProblemLossTask(with_noise=True))
    task_repository.add_task(ExampleFirst2ProblemLossWithWeightTask(with_noise=True))

    task_repository.add_task(ExampleSecond2ProblemLossTaskWithWeight(with_noise=True))
    task_repository.add_task(ExampleSecond2ProblemLossTask(with_noise=True))


def configure_solver(model_with_optimization: bool, wang_configuration: bool):
    model_with_optimization_configuration = ModelWithOptimizationConfiguration(epochs=10, number_of_trials=2) \
        if model_with_optimization else None
    model_configuration = ModelConfiguration()
    wang_params = WangParams(hidden_dim=10, activation_function='sigmoid') if wang_configuration else None
    model_configuration.configure(model_with_optimization=model_with_optimization_configuration,
                                  wang_configuration=wang_params)


def get_tests_repository_with_all_equations():
    task_repository = TasksRepository()

    add_all_first_degree(task_repository)
    add_all_second_degree(task_repository)

    return task_repository
