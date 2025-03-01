from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService

from tasks.ai.article.examples.ArticleExamplesImport import *

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)

    task_repository.add_task(ThirdbProblemSimpleTask())
    task_repository.add_task(ThirdbProblemSimpleLossTask())

    task_service.solve(20000)

