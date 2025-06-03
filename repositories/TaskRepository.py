from objects.TaskData import TaskData


class TasksRepository:
    def __init__(self):
        self.__tasks = []

    def add_task(self, task: TaskData):
        self.__tasks.append(task)

    def get_tasks(self):
        return self.__tasks
