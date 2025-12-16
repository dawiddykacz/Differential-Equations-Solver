import numpy
from tqdm import tqdm
import copy

from repositories.TaskRepository import TasksRepository
from objects.space.Space import Space
from objects.TaskData import TaskData
from plots.ChoosePlot import ChoosePlot
from objects.plot.PlotData import PlotData
from objects.functions.error.AbsError import AbsError
from objects.functions.error.PercentError import PercentError
from statistics import mean

_equations_amount = 1


def set_equations_amount(equations_amount: int):
    global _equations_amount
    if equations_amount <= 0:
        raise "equations_amount must be greater than zero"
    _equations_amount = equations_amount


class TaskService:
    def __init__(self, task_repository: TasksRepository):
        self.__task_repository = task_repository
        import time
        self.__ms = round(time.time() * 1000)
        self.__task_dict = dict()
        self.__epochs = 0
        self.__error_messages = []

    def solve(self, epochs: int, multiply_space: int = 10):
        print(f"solving {epochs} epochs")
        all_tasks = self.__task_repository.get_tasks()
        for i in tqdm(range(len(all_tasks))):
            task = all_tasks[i]
            self.__run_task(task, epochs, multiply_space)
        self.__epochs += epochs

    def run_solution(self, equation, epoch):
        ai_solution = equation.get_solution_function()
        ai_solution.solve(epoch)

    def __run_task(self, task: TaskData, epoch: int, multiply_space: int):
        equation = task.get_equation()
        plot_title = f"{task.get_task_name()} epoches: {epoch}"
        test_space = task.get_space_range().split(multiply_space)

        equations = []
        equations_amount = _equations_amount

        for i in range(equations_amount):
            print(f'{i + 1} / {equations_amount}')
            eq = copy.deepcopy(equation)
            equations.append(eq)
            self.run_solution(eq, epoch)

        y = copy.deepcopy(equations[0].get_solution_function().calculate_as_numpy(test_space)) * 0
        abs_error = copy.deepcopy(equations[0].get_solution_function().calculate_as_numpy(test_space)) * 0
        max_error = 0
        avg_error = 0
        min_error = 0
        percent_error = copy.deepcopy(equations[0].get_solution_function().calculate_as_numpy(test_space)) * 0
        variables_array = copy.deepcopy(equations[0].get_solution_function().get_trainable_variables_array())
        for i in range(len(variables_array)):
            for j in range(len(variables_array[i])):
                variables_array[i][j] = 0
        loss_array = copy.deepcopy(equations[0].get_solution_function().get_loss_array())
        for i in range(len(loss_array)):
            loss_array[i] = 0

        for eq in equations:
            y += eq.get_solution_function().calculate_as_numpy(test_space)
            if variables_array is not None:
                variables_array_temp = copy.deepcopy(eq.get_solution_function().get_trainable_variables_array())
                for i in range(len(variables_array)):
                    for j in range(len(variables_array[i])):
                        variables_array[i][j] += variables_array_temp[i][j]
            if loss_array is not None:
                loss_array += copy.deepcopy(eq.get_solution_function().get_loss_array())

        y /= equations_amount
        if loss_array is not None and len(loss_array) > 0:
            loss_array /= equations_amount
            space = Space([numpy.linspace(1, epoch, epoch)])
            choose_plot = ChoosePlot(space, loss_array,
                                     self.__get_plot_path(task.get_task_name(), "Convergence"),
                                     PlotData("Convergence", ["epoch", "loss"]))
            choose_plot.choose().plot()

            threshold = (loss_array[0] - loss_array[len(loss_array) - 1]) / 10
            intervals = self.__find_small_change_intervals(loss_array, threshold)
            i = 0
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
                    i += 1
        if variables_array is not None and len(variables_array) > 0:
            for i in range(len(variables_array)):
                for j in range(len(variables_array[i])):
                    variables_array[i][j] /= equations_amount
            space = Space([numpy.linspace(1, epoch, epoch)])
            for i in range(len(variables_array)):
                variable = variables_array[i]
                choose_plot = ChoosePlot(space, variable,
                                         self.__get_plot_path(task.get_task_name(), f"Trainable variable {i}"),
                                         PlotData(f"Trainable variable {i}", ["epoch", "value"]))
                choose_plot.choose().plot()

        choose_plot = ChoosePlot(test_space, y, self.__get_plot_path(task.get_task_name(), "Ai Solution"),
                                 PlotData(f"Ai solution {plot_title}"))
        choose_plot.choose().plot()

        exact_solution = equations[0].get_exact_solution()
        max_percent_error = None
        if exact_solution is not None:
            exact_y = exact_solution.calculate_as_numpy(test_space)
            choose_plot = ChoosePlot(test_space, exact_y,
                                     self.__get_plot_path(task.get_task_name(), "Exact Solution"),
                                     PlotData(f"Exact solution {task.get_task_name()}"))
            choose_plot.choose().plot()
            exact_y_error = numpy.abs(numpy.max(exact_y))

            labels = []
            d = test_space.get_dimension()
            if d == 1:
                labels = ["x", "Error"]
            if d == 2:
                labels = ["x", "y", "Error"]
            for eq in equations:
                ai_solution = eq.get_solution_function()
                abs_error_function = AbsError(ai_solution, exact_solution)
                abs_error = abs_error_function.calculate(test_space)

                percent_error_function = PercentError(ai_solution, exact_solution)
                percent_error_1 = percent_error_function.calculate(test_space)
                if percent_error_1 is None:
                    percent_error = None
                else:
                    percent_error += percent_error_1
                max_error += self.__get_max_error(abs_error_function.calculate(test_space))
                avg_error += self.__get_avg_error(abs_error_function.calculate(test_space))
                min_error += self.__get_min_error(abs_error_function.calculate(test_space))

            max_error /= equations_amount
            avg_error /= equations_amount
            min_error /= equations_amount

            choose_plot = ChoosePlot(test_space, abs_error,
                                     self.__get_plot_path(task.get_task_name(), "Absolute error"),
                                     PlotData(f"Absolute error {plot_title}", labels))
            choose_plot.choose().plot()
            max_percent_error = numpy.max(abs_error) / exact_y_error * 100
            if percent_error is not None:
                percent_error /= equations_amount
                choose_plot = ChoosePlot(test_space, percent_error,
                                         self.__get_plot_path(task.get_task_name(), "Percent error"),
                                         PlotData(f"Error % {plot_title}",
                                                  ["x", "Error (%)"]))
                choose_plot.choose().plot()
                max_percent_error = numpy.max(percent_error)
                weight = task.get_weight()
                if weight is not None:
                    k = str(weight)
                    task_name = task.get_task_name_simple()
                    self.__task_dict.setdefault(f'{task_name} (max)', dict())
                    self.__task_dict.setdefault(f'{task_name} (min)', dict())
                    self.__task_dict.setdefault(f'{task_name} (avg)', dict())

                    self.__task_dict[f'{task_name} (max)'][k] = max_error
                    self.__task_dict[f'{task_name} (min)'][k] = avg_error
                    self.__task_dict[f'{task_name} (avg)'][k] = min_error
        if max_percent_error is not None:
            error_message = f"{task.get_task_name()} epoches: {epoch} max error ~ {max_percent_error}%"
            self.__error_messages.append(error_message)

    def get_error_messages(self):
        return self.__error_messages

    def get_task_dict(self):
        return self.__task_dict

    def __get_max_error(self, arr: []):
        a = []
        for b in arr:
            a.append(max(b))
        return max(a)

    def __get_avg_error(self, arr: []):
        a = []
        for b in arr:
            a.append(mean(b))
        return mean(a)

    def __get_min_error(self, arr: []):
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
