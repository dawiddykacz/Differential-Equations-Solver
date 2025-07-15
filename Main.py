from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService
from services.WeightPlotService import WeightPlotService

from tasks.ai.article.examples.ArticleExamplesImport import *

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)
    weight_plot_service = WeightPlotService(task_service.get_ms())


    task_repository.add_task(SeventhProblemSimpleTask())

    #task_service.solve(300)
    task_service.solve(3000)
    weight_plot_service.plots(task_service.get_task_dict(),task_service.get_epochs())
