from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService

from tasks.ai.article.examples.ArticleExamplesImport import *
from solvers.AISolver import set_learning_rate
from services.TaskService import set_equations_amount


def run_all(learning_rate: float):
    set_learning_rate(learning_rate)
    set_equations_amount(20)

    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    for i in range(0, 100):
        a = i 
        if a <= 0:
            a = 1

        task_repository.add_task(FirstProblemLossTask(a))
        # task_repository.add_task(SecondProblemLossTask(a))
        # task_repository.add_task(ThirdProblemLossTask(a))
        # task_repository.add_task(ThirdbProblemLossTask(a))
        # task_repository.add_task(FifthProblemLossTask(a))
        # task_repository.add_task(SixthProblemLossTask(a))
        # task_repository.add_task(SeventhProblemLossTask(a))

    # task_repository.add_task(FifthProblemWithDistanceFunctionTask())
    # task_repository.add_task(FifthProblemLossWithPointTask())
    # task_repository.add_task(FirstProblemSimpleTask())
    # task_repository.add_task(SecondProblemSimpleTask())
    # task_repository.add_task(ThirdProblemSimpleTask())
    # task_repository.add_task(ThirdbProblemSimpleTask())
    # task_repository.add_task(FifthProblemSimpleTask())
    # task_repository.add_task(SixthProblemSimpleTask())
    # task_repository.add_task(SeventhProblemSimpleTask())
    #
    task_repository.add_task(FirstProblemLossWithWeightTask())
    # task_repository.add_task(SecondProblemLossWithWeightTask())
    # task_repository.add_task(ThirdbProblemLossWithWeightTask())
    # task_repository.add_task(ThirdProblemLossWithWeightTask())
    # task_repository.add_task(FifthProblemLossWithWeightTask())
    # task_repository.add_task(SixthProblemLossWithWeightTask())
    # task_repository.add_task(SeventhProblemLossWithWeightTask())

    task_service.solve(5)
    weight_plot_service.plots(task_service.get_task_dict(), task_service.get_epochs())

    error_messages = task_service.get_error_messages()
    if error_messages is not None:
        for error_message in error_messages:
            print(error_message)


if __name__ == '__main__':
    run_all(0.1)
