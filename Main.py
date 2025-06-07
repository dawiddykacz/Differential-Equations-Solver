from repositories.TaskRepository import TasksRepository
from services.TaskService import TaskService

from tasks.ai.article.examples.ArticleExamplesImport import *

if __name__ == '__main__':
    task_repository = TasksRepository()
    task_service = TaskService(task_repository)

    task_repository.add_task(FirstProblemSimpleTask())
    task_repository.add_task(FirstProblemLossTask())
    task_repository.add_task(FirstProblemLossWithWeightTask())
    task_repository.add_task(SecondProblemSimpleTask())
    task_repository.add_task(SecondProblemLossTask())
    task_repository.add_task(ThirdProblemSimpleTask())
    task_repository.add_task(ThirdProblemLossTask())
    task_repository.add_task(ThirdbProblemSimpleTask())
    task_repository.add_task(ThirdbProblemLossTask())
    task_repository.add_task(FourthProblemSimpleTask())

    task_service.solve(60000)

