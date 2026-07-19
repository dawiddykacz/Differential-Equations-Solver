import unittest
from parameterized import parameterized

from services.TaskService import set_equations_amount, TaskService
from services.WeightPlotService import WeightPlotService
from solvers.AISolver import set_learning_rate
from tests.TestHelper import get_tests_repository_with_all_equations, configure_solver


class EquationsTest(unittest.TestCase):
    @parameterized.expand([
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ])
    def test_should_run_all_cases(self, model_with_optimization, wang_configuration):
        configure_solver(model_with_optimization=model_with_optimization, wang_configuration=wang_configuration)
        set_learning_rate(0.1)
        set_equations_amount(1)

        task_repository = get_tests_repository_with_all_equations()
        task_service = TaskService(task_repository)
        weight_plot_service = WeightPlotService(task_service.get_ms())

        task_service.solve(10)

        weight_plot_service.plots(task_service.get_task_dict(), task_service.get_epochs())

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
