from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService

from tasks.ai.article.examples.ArticleExamplesImport import *

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)

    for i in range(1,10):
        task_repository.add_task(FirstProblemLossTask(i*10))
    task_repository.add_task(FifthProblemSimpleTask())

    task_service.solve(100)
    print(task_service.get_task_dict())
