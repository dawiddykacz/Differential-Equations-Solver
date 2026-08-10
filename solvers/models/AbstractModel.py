import copy

import tensorflow

from objects.TrainableVariables import TrainableVariables
from objects.functions.loss.LossFunction import LossFunction


class AbstractModel(tensorflow.keras.Model):
    def __init__(self, loss, trainable_variables: TrainableVariables, dense_list, optimizer=None, custom_layers=None):
        super(AbstractModel, self).__init__()
        if custom_layers is None:
            custom_layers = []
        self.loss_tracker = tensorflow.keras.metrics.Mean(name="loss")
        self.dense_list = tensorflow.keras.Sequential(dense_list)
        self.out_dense = tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')
        self._loss = loss
        self._custom_trainable_variables = trainable_variables
        self.optimizer = optimizer
        self._custom_layers = custom_layers

    def call(self, inputs):
        x = self.dense_list(inputs)
        output = self.out_dense(x)
        return output

    @tensorflow.function
    def train_step(self, data=None):
        with tensorflow.GradientTape(persistent=True) as tape:
            loss = self._loss()
            current_loss = loss['loss']
            loss_pde = loss['loss_pde']
            conditions = loss['conditions']
            conditions_data = loss['conditions_data']

        variables_to_train = self.trainable_variables + self._custom_trainable_variables.get_variables()
        grads = tape.gradient(current_loss, variables_to_train)
        self.optimizer.apply_gradients(zip(grads, variables_to_train))

        grad_data = []
        grad_pde = []
        grad_bc = []

        layers = list(self.dense_list.layers) + [self.out_dense]
        for layer in layers:
            weights = layer.kernel

            grad_data.append(
                tape.gradient(tensorflow.convert_to_tensor(conditions_data, dtype=tensorflow.float64), weights)
            )

            grad_pde.append(
                tape.gradient(tensorflow.convert_to_tensor(loss_pde, dtype=tensorflow.float64), weights)
            )

            grad_bc.append(
                tape.gradient(tensorflow.convert_to_tensor(conditions, dtype=tensorflow.float64), weights)
            )

        del tape

        grad_data = [
            tensorflow.reshape(g, [-1]) if g is not None else None
            for g in grad_data
        ]

        grad_pde = [
            tensorflow.reshape(g, [-1]) if g is not None else None
            for g in grad_pde
        ]

        grad_bc = [
            tensorflow.reshape(g, [-1]) if g is not None else None
            for g in grad_bc
        ]

        return {
            'loss': current_loss,
            'grad_data_mean': LossFunction.mean_abs_grads(grad_data),
            'grad_pde_max': LossFunction.max_abs_grads(grad_pde),
            'grad_bc_mean': LossFunction.mean_abs_grads(grad_bc),
        }

    @tensorflow.function
    def estimate_stiffness(self, num_iters=3):
        """
        Estymuje największą wartość własną Hessjanu (sztywność) za pomocą
        iloczynów Hessian-Vector Product (HVP) i metody potęgowej.
        """
        variables_to_track = self.trainable_variables + self._custom_trainable_variables.get_variables()

        # Inicjalizacja losowego wektora v o tych samych wymiarach co wagi modelu (wymagane float64)
        v_list = [tensorflow.random.normal(shape=w.shape, dtype=tensorflow.float64) for w in variables_to_track]

        for _ in range(num_iters):
            # Normalizacja wektora
            norm = tensorflow.sqrt(tensorflow.add_n([tensorflow.reduce_sum(tensorflow.square(v)) for v in v_list]))
            v_list = [v / norm for v in v_list]

            # Obliczenie HVP
            with tensorflow.GradientTape() as outer_tape:
                with tensorflow.GradientTape() as inner_tape:
                    loss_dict = self._loss()
                    total_loss = loss_dict['loss']

                # Pierwsza pochodna
                grads = inner_tape.gradient(total_loss, variables_to_track)
                # Iloczyn skalarny
                grad_v_dot = tensorflow.add_n(
                    [tensorflow.reduce_sum(g * v) for g, v in zip(grads, v_list) if g is not None])

            # Druga pochodna (H * v)
            Hv_list = outer_tape.gradient(grad_v_dot, variables_to_track)

            # Zapobiegawcze filtrowanie None (jeśli jakieś wagi nie mają wpływu na loss)
            Hv_list = [hv if hv is not None else tensorflow.zeros_like(v) for hv, v in zip(Hv_list, v_list)]
            v_list = Hv_list

        # Iloraz Rayleigha (Rayleigh quotient) do wyciągnięcia lambda_max
        norm = tensorflow.sqrt(tensorflow.add_n([tensorflow.reduce_sum(tensorflow.square(v)) for v in v_list]))
        v_list_normalized = [v / norm for v in v_list]

        with tensorflow.GradientTape() as outer_tape:
            with tensorflow.GradientTape() as inner_tape:
                loss_dict = self._loss()
                total_loss = loss_dict['loss']
            grads = inner_tape.gradient(total_loss, variables_to_track)
            grad_v_dot = tensorflow.add_n(
                [tensorflow.reduce_sum(g * v) for g, v in zip(grads, v_list_normalized) if g is not None])

        Hv_final = outer_tape.gradient(grad_v_dot, variables_to_track)
        Hv_final = [hv if hv is not None else tensorflow.zeros_like(v) for hv, v in zip(Hv_final, v_list_normalized)]

        lambda_max = tensorflow.add_n([tensorflow.reduce_sum(v * Hv) for v, Hv in zip(v_list_normalized, Hv_final)])
        return lambda_max

    @property
    def metrics(self):
        return [
            self.loss_tracker
        ]

    def __deepcopy__(self, memo):
        cloned_layers = [
            layer.__class__.from_config(layer.get_config())
            for layer in self.dense_list.layers
        ]
        cloned_custom_layers = [
            layer.__class__.from_config(layer.get_config())
            for layer in self._custom_layers
        ]
        new_model = AbstractModel(loss=copy.deepcopy(self._loss, memo),
                                  trainable_variables=copy.deepcopy(self._custom_trainable_variables, memo),
                                  optimizer=copy.deepcopy(self.optimizer, memo),
                                  dense_list=cloned_layers,
                                  custom_layers=cloned_custom_layers)

        memo[id(self)] = new_model

        return new_model
