from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService

from tasks.ai.article.examples.ArticleExamplesImport import *

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())

    task_repository.add_task(FirstProblemLossTask(1))
    for i in range(1,10):
        task_repository.add_task(FirstProblemLossTask(i*10))
    task_repository.add_task(FifthProblemSimpleTask())

    task_service.solve(10000)
    weight_plot_service.plots(task_service.get_task_dict(),task_service.get_epochs())
