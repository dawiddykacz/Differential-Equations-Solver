import tensorflow
import numpy

from solvers.models.ChooseModel import ModelParams, choose_model
from objects.space.Space import Space
from objects.functions.loss.LossFunction import LossFunction
from objects.TrainableVariables import TrainableVariables
from solvers.models.ModelConfiguration import ModelConfiguration

_learning_rate = 0.1
model_configuration = ModelConfiguration()


def set_learning_rate(learning_rate: float = 0.1):
    global _learning_rate
    if learning_rate <= 0.0:
        raise ValueError("Learning rate must be greater than 0.")
    _learning_rate = learning_rate


class AISolver:
    def __init__(self, space: Space, solution_function, loss_function: LossFunction,
                 trainable_variables: TrainableVariables = TrainableVariables(),
                 non_trainable_variables: TrainableVariables = TrainableVariables(), plots: bool = True,
                 calculate_as_numpy=None):
        self.__points = space.get_points_to_neural_network()
        self.__solution_function = solution_function
        self.__loss_function = loss_function
        self.__plots = plots
        self.__calculate_as_numpy = calculate_as_numpy

        if trainable_variables is None:
            self.__trainable_variables = TrainableVariables()
        else:
            self.__trainable_variables = trainable_variables

        if non_trainable_variables is None:
            self.__non_trainable_variables = TrainableVariables()
        else:
            self.__non_trainable_variables = non_trainable_variables

        model_params = ModelParams(loss=self.current_loss,
                                   trainable_variables=trainable_variables,
                                   optimizer=model_configuration.get_optimizer(_learning_rate))

        self.__neural_network = choose_model(params=model_params,
                                             with_optimization=model_configuration.can_optimize(),
                                             wang_params=model_configuration.wang_configuration)

        self.__trainable_plot = []

        for _ in self.__trainable_variables.get_variables():
            self.__trainable_plot.append([])

        self.__non_trainable_plot = []
        self.__y_by_epoch = []

        for _ in self.__non_trainable_variables.get_variables():
            self.__non_trainable_plot.append([])

        self.__loss_array = []
        self.__loss_pde_array = []
        self.__loss_conditions_array = []
        self.__loss_conditions_data_array = []

        self.__grad_pde_max_array = []
        self.__grad_bc_max_array = []
        self.__grad_data_max_array = []

        self.__grad_pde_mean_array = []
        self.__grad_data_mean_array = []
        self.__grad_bc_mean_array = []
        self.__network_stiffness_array = []

        if len(self.__points) > 1:
            self.__inputs = tensorflow.concat(self.__points, axis=1)
        else:
            self.__inputs = self.__points[0]

    def calculate(self, *variables):
        inputs = tensorflow.concat(variables, axis=1)
        return self.__neural_network(inputs)

    def current_loss(self):
        return self.__loss_function.calculate(self.__solution_function, *self.__points)

    def solve(self, epochs: int, test_points):
        self.__neural_network.init(self.__inputs)

        # Tablice do zbierania danych diagnostycznych
        self.diagnostics_lambda_max = []
        self.diagnostics_grad_ratio = []

        # NOWE: Parametry konfiguracyjne Curriculum Learning
        diagnostic_interval = 500
        recovery_steps = 500  # Ile iteracji zajmuje powrót wagi PDE do 1.0
        current_recovery_step = 0
        in_recovery = False
        stiffness_threshold = 100.0  # Krytyczna wartość lambda_max

        before_loss = self.current_loss()["loss"]
        for i in range(epochs):
            loss = self.__neural_network.train_step()
            loss_dict = loss
            current_loss = loss["loss"]
            loss_error = tensorflow.abs((current_loss - before_loss) / before_loss + 1e-8)
            before_loss = current_loss

            lambda_max = self.__neural_network.estimate_stiffness(num_iters=3).numpy()

            if self.__plots:
                y = self.__calculate_as_numpy(test_points)
                self.__y_by_epoch.append(y)

                self.__network_stiffness_array.append(lambda_max)
                self.__loss_array.append(current_loss.numpy())
                self.__loss_pde_array.append(loss['loss_pde'].numpy())
                self.__loss_conditions_array.append(loss['loss_conditions'].numpy())
                self.__loss_conditions_data_array.append(loss['loss_conditions_data'].numpy())

                self.__grad_pde_max_array.append(loss['grad_pde_max'].numpy())
                self.__grad_bc_max_array.append(loss['grad_bc_max'].numpy())
                self.__grad_data_max_array.append(loss['grad_data_max'].numpy())

                self.__grad_pde_mean_array.append(loss['grad_pde_mean'].numpy())
                self.__grad_data_mean_array.append(loss['grad_data_mean'].numpy())
                self.__grad_bc_mean_array.append(loss['grad_bc_mean'].numpy())

                for j in range(len(self.__trainable_plot)):
                    self.__trainable_plot[j].append(self.__trainable_variables.get_variables()[j].numpy())
                for j in range(len(self.__non_trainable_plot)):
                    self.__non_trainable_plot[j].append(self.__non_trainable_variables.get_variables()[j].numpy())
            self.__loss_function.recalculate_weights(loss, loss_error)

            if model_configuration.optimize_condition_weight:
                # ==========================================
                # NOWE: Logika powrotu ze "Znieczulenia"
                # ==========================================
                if in_recovery:
                    # Liniowy wzrost wagi od 0.01 do 1.0
                    progress = current_recovery_step / recovery_steps
                    new_weight = 0.01 + progress * (1.0 - 0.01)
                    self.__loss_function.assign_weights([new_weight])

                    current_recovery_step += 1
                    if current_recovery_step >= recovery_steps:
                        in_recovery = False
                        self.__loss_function.assign_weights([1.0])
                        print(f"Epoka {i:05d} | [SYSTEM DIAGNOSTYCZNY] Zakończono rekonwalescencję. Waga PDE = 1.0.")

                # ==========================================
                # NOWE: Moduł Diagnostyczny (Uruchamiany poza rekonwalescencją)
                # ==========================================
                if i % diagnostic_interval == 0 and not in_recovery:
                    # 1. Analiza sztywności
                    self.diagnostics_lambda_max.append(lambda_max)

                    # 2. Analiza imbalansu gradientów
                    grad_pde = loss_dict.get('grad_pde_max', 0.0)
                    grad_bc = loss_dict.get('grad_bc_mean', 0)
                    grad_data = loss_dict.get('grad_data_mean', 1e-8)
                    if isinstance(grad_pde, tensorflow.Tensor): grad_pde = grad_pde.numpy()

                    if grad_bc != 0:
                        if isinstance(grad_bc, tensorflow.Tensor): grad_bc = grad_bc.numpy()

                        grad_ratio = grad_pde / (grad_bc + 1e-8)
                        self.diagnostics_grad_ratio.append(grad_ratio)

                        print(
                            f"Epoka {i:05d} | Loss: {current_loss:.4e} | Grad Ratio (PDE/BC): {grad_ratio:.2f} | Sztywność (lambda_max): {lambda_max:.2f}")
                    else:
                        if isinstance(grad_data, tensorflow.Tensor): grad_data = grad_data.numpy()

                        grad_ratio = grad_pde / (grad_data + 1e-8)
                        self.diagnostics_grad_ratio.append(grad_ratio)

                        print(
                            f"Epoka {i:05d} | Loss: {current_loss:.4e} | Grad Ratio (PDE/BC): {grad_ratio:.2f} | Sztywność (lambda_max): {lambda_max:.2f}")

                    # 3. INTERWENCJA SYSTEMU (Aplikacja "Znieczulenia")
                    if lambda_max > stiffness_threshold or grad_ratio > 10000.0:
                        print(f"  [UWAGA] Sztywność przekroczyła próg ({lambda_max:.2f} > {stiffness_threshold}).")
                        print("  [AKCJA] Aplikuję 'znieczulenie' PDE. Waga lambda_pde zredukowana do 0.01.")
                        self.__loss_function.assign_weights([0.01])
                        in_recovery = True
                        current_recovery_step = 0

    def get_loss_array(self):
        if self.__plots:
            return numpy.array(self.__loss_array)
        return None

    def get_network_stiffness_array(self):
        if self.__plots:
            return numpy.array(self.__network_stiffness_array)
        return None

    def get_loss_pde_array(self):
        if self.__plots:
            return numpy.array(self.__loss_pde_array)
        return None

    def get_loss_conditions_array(self):
        if self.__plots:
            return numpy.array(self.__loss_conditions_array)
        return None

    def get_loss_conditions_data_array(self):
        if self.__plots:
            return numpy.array(self.__loss_conditions_data_array)
        return None

    def get_grad_pde_max_array(self):
        if self.__plots:
            return numpy.array(self.__grad_pde_max_array)
        return None

    def get_grad_bc_max_array(self):
        if self.__plots:
            return numpy.array(self.__grad_bc_max_array)
        return None

    def get_grad_data_max_array(self):
        if self.__plots:
            return numpy.array(self.__grad_data_max_array)
        return None

    def get_grad_pde_mean_array(self):
        if self.__plots:
            return numpy.array(self.__grad_pde_mean_array)
        return None

    def get_grad_data_mean_array(self):
        if self.__plots:
            return numpy.array(self.__grad_data_mean_array)
        return None

    def get_grad_bc_mean_array(self):
        if self.__plots:
            return numpy.array(self.__grad_bc_mean_array)
        return None

    def get_trainable_variables_array(self):
        if self.__plots:
            return self.__trainable_plot
        return None

    def get_non_trainable_variables_array(self):
        if self.__plots:
            return self.__non_trainable_plot
        return None

    def get_y_by_epoch(self):
        if self.__plots:
            return self.__y_by_epoch
        return None
