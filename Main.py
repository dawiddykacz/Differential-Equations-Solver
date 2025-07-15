from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService

from tasks.ai.article.examples.ArticleExamplesImport import *

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    task_repository.add_task(FirstProblemSimpleTask())
    task_repository.add_task(FirstProblemLossWithWeightTask())
    for i in range(0,10):
        a = i * 10
        if a <= 0:
            a = 1

        task_repository.add_task(FirstProblemLossTask(a))
        task_repository.add_task(SecondProblemLossTask(a))
        task_repository.add_task(ThirdProblemLossTask(a))
        task_repository.add_task(ThirdbProblemLossTask(a))

    task_repository.add_task(SecondProblemSimpleTask())
    task_repository.add_task(ThirdProblemSimpleTask())
    task_repository.add_task(ThirdbProblemSimpleTask())

    task_repository.add_task(SecondProblemLossWithWeightTask())
    task_repository.add_task(ThirdProblemLossWithWeightTask())
    task_repository.add_task(ThirdProblemLossWithWeightTask())
    task_repository.add_task(FifthProblemSimpleTask())
    task_repository.add_task(SixthProblemSimpleTask())
    task_repository.add_task(SeventhProblemSimpleTask())

    #task_service.solve(300)
    task_service.solve(10000)
    weight_plot_service.plots(task_service.get_task_dict(),task_service.get_epochs())
