from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService

from tasks.ai.article.examples.ArticleExamplesImport import *
from solvers.AISolver import set_learning_rate

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    # for i in range(0,10):
    #     a = i * 10
    #     if a <= 0:
    #         a = 1
    #
    #     task_repository.add_task(FirstProblemLossTask(a))
    #     task_repository.add_task(SecondProblemLossTask(a))
    #     task_repository.add_task(ThirdProblemLossTask(a))
    #     task_repository.add_task(ThirdbProblemLossTask(a))
    #
    # task_repository.add_task(FirstProblemSimpleTask())
    # task_repository.add_task(SecondProblemSimpleTask())
    # task_repository.add_task(ThirdProblemSimpleTask())
    # task_repository.add_task(ThirdbProblemSimpleTask())
    # task_repository.add_task(SixthProblemSimpleTask())
    # task_repository.add_task(SeventhProblemSimpleTask())
    #
    # task_repository.add_task(FirstProblemLossWithWeightTask())
    # task_repository.add_task(SecondProblemLossWithWeightTask())
    # task_repository.add_task(ThirdbProblemLossWithWeightTask())
    # task_repository.add_task(ThirdProblemLossWithWeightTask())
    #task_repository.add_task(FifthProblemSimpleTask())
    set_learning_rate()

    task_repository.add_task(FifthProblemSimpleTask())
    task_repository.add_task(SeventhProblemSimpleTask())
    task_repository.add_task(SixthProblemSimpleTask())

    task_repository.add_task(FifthProblemLossTask())
    task_repository.add_task(SixthProblemLossTask())
    task_repository.add_task(SeventhProblemLossTask())

    task_service.solve(1000)
    weight_plot_service.plots(task_service.get_task_dict(),task_service.get_epochs())
