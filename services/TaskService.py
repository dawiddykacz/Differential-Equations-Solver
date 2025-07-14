import numpy
from tqdm import tqdm

from repositories.TaskRepository import TasksRepository
from objects.space.Space import Space
from objects.TaskData import TaskData
from plots.ChoosePlot import ChoosePlot
from objects.plot.PlotData import PlotData
from objects.functions.error.AbsError import AbsError
from objects.functions.error.PercentError import PercentError
from statistics import mean


class TaskService:
    def __init__(self, task_repository: TasksRepository):
        self.__task_repository = task_repository
        import time
        self.__ms = round(time.time() * 1000)
        self.task_dict = dict()
        self.__epochs = 0

    def solve(self, epochs: int, multiply_space: int = 10):
        all_tasks = self.__task_repository.get_tasks()
        for i in range(len(all_tasks)):
            print(f'{i+1}/{len(all_tasks)}')
            task = all_tasks[i]
            self.__run_task(task, epochs, multiply_space)
        self.__epochs += epochs

    def __run_task(self, task: TaskData, epoch: int, multiply_space: int):
        equation = task.get_equation()
        plot_title = f"{task.get_task_name()} epoches: {epoch}"

        ai_solution = equation.get_solution_function()
        ai_solution.solve(epoch)

        test_space = task.get_space_range().split(multiply_space)
        y = ai_solution.calculate_as_numpy(test_space)
        choose_plot = ChoosePlot(test_space, y, self.__get_plot_path(task.get_task_name(), "Ai Solution"),
                                 PlotData(f"Ai solution {plot_title}"))
        choose_plot.choose().plot()

        exact_solution = equation.get_exact_solution()
        if exact_solution is not None:

            choose_plot = ChoosePlot(test_space, exact_solution.calculate_as_numpy(test_space),
                                     self.__get_plot_path(task.get_task_name(), "Exact Solution"),
                                     PlotData(f"Exact solution {task.get_task_name()}"))
            choose_plot.choose().plot()

            abs_error_function = AbsError(ai_solution, exact_solution)

            choose_plot = ChoosePlot(test_space, abs_error_function.calculate(test_space),
                                     self.__get_plot_path(task.get_task_name(), "Absolute error"),
                                     PlotData(f"Absolute error {plot_title}", ["x", "Error"]))
            choose_plot.choose().plot()

            percent_error_function = PercentError(ai_solution, exact_solution)
            percent_error = percent_error_function.calculate(test_space)
            if percent_error is not None:
                choose_plot = ChoosePlot(test_space, percent_error,
                                         self.__get_plot_path(task.get_task_name(), "Percent error"),
                                         PlotData(f"Error % {plot_title}",
                                                  ["x", "Error (%)"]))
                choose_plot.choose().plot()

            weight = task.get_weight()
            if weight is not None:
                k = str(weight)
                task_name = task.get_task_name_simple()
                self.task_dict.setdefault(f'{task_name} (max)',dict())
                self.task_dict.setdefault(f'{task_name} (min)',dict())
                self.task_dict.setdefault(f'{task_name} (avg)',dict())

                self.task_dict[f'{task_name} (max)'][k] = self.__get_max_error(abs_error_function.calculate(test_space))
                self.task_dict[f'{task_name} (min)'][k] = self.__get_min_error(abs_error_function.calculate(test_space))
                self.task_dict[f'{task_name} (avg)'][k] = self.__get_avg_error(abs_error_function.calculate(test_space))
        variables_array = ai_solution.get_trainable_variables_array()
        if variables_array is not None:
            space = Space([numpy.linspace(1, epoch, epoch)])
            for i in range(len(variables_array)):
                variable = variables_array[i]
                choose_plot = ChoosePlot(space, variable,
                                         self.__get_plot_path(task.get_task_name(), f"Trainable variable {i}"),
                                         PlotData(f"Trainable variable {i}", ["epoch", "value"]))
                choose_plot.choose().plot()

        loss_array = ai_solution.get_loss_array()
        if loss_array is not None:
            space = Space([numpy.linspace(1, epoch, epoch)])
            choose_plot = ChoosePlot(space, loss_array,
                                     self.__get_plot_path(task.get_task_name(), "Convergence"),
                                     PlotData("Convergence", ["epoch", "loss"]))
            choose_plot.choose().plot()

            threshold = (loss_array[0] - loss_array[len(loss_array) - 1]) / 10
            intervals = self.__find_small_change_intervals(loss_array, threshold)

            i=0
            while threshold > loss_array[len(loss_array) - 1]:
                for interval in intervals:
                    start = interval[0]
                    end = epoch
                    if len(interval) >= 2:
                        end = interval[1]

                    loss_sub_array = loss_array[start:end]

                    space = Space([numpy.linspace(start, end, end - start)])
                    choose_plot = ChoosePlot(space, loss_sub_array,
                                             self.__get_plot_path(task.get_task_name(), f'Convergence-{i}'),
                                             PlotData("Convergence", ["epoch", "loss"]))
                    choose_plot.choose().plot()

                threshold = threshold / 10
                intervals = self.__find_small_change_intervals(loss_array, threshold)
                i+=1

    def get_task_dict(self):
        return self.task_dict

    def __get_max_error(self, arr : []):
        a = []
        for b in arr:
            a.append(max(b))
        return max(a)

    def __get_avg_error(self, arr : []):
        a = []
        for b in arr:
            a.append(mean(b))
        return mean(a)

    def __get_min_error(self, arr : []):
        a = []
        for b in arr:
            a.append(min(b))
        return min(a)

    def __find_small_change_intervals(self, array, change_threshold):
        intervals = []

        i = 0
        while i < len(array) - 1:
            start = i
            end = len(array) - 1

            if start != 0 and abs(array[start] - array[end]) <= change_threshold:

                if end - start >= 4:
                    intervals.append((start, end))
                    return intervals

            i += 1

        return intervals

    def get_ms(self):
        return self.__ms

    def get_epochs(self):
        return self.__epochs

    def __get_plot_path(self, task_name: str, plot_name: str):
        return f'plot/{self.__ms}/{task_name}/{plot_name}.png'
