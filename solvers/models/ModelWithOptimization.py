import tensorflow
import keras_tuner
from solvers.models.AbstractModel import AbstractModel
from objects.TrainableVariables import TrainableVariables

import keras_tuner

from solvers.models.ModelConfiguration import ModelConfiguration

model_configuration = ModelConfiguration()


class MyRandomSearch(keras_tuner.RandomSearch):

    def __init__(self, hypermodel, **kwargs):
        super().__init__(hypermodel, **kwargs)

        self._first_trial = True

    def search(self, *args, **kwargs):
        if self._first_trial:
            self._first_trial = False

            hp = keras_tuner.HyperParameters()

            hp.values = {
                "num_layers": 1,
                "units_0": 10,
                "activations_0": "sigmoid",
                "learning_rate": 0.1,
            }

            trial = keras_tuner.engine.trial.Trial(
                hyperparameters=hp,
                trial_id="manual_default"
            )

            self.oracle.trials[trial.trial_id] = trial

            self.run_trial(trial, *args, **kwargs)

        return super().search(*args, **kwargs)


class ModelWithOptimization(AbstractModel):
    def __init__(self, wrapper, loss, trainable_variables: TrainableVariables, hp):
        self.__basic_loss = loss
        self.__wrapper = wrapper

        num_layers = hp.Int('num_layers',
                            min_value=model_configuration.model_with_optimization.number_of_layers_range.start,
                            max_value=model_configuration.model_with_optimization.number_of_layers_range.end,
                            default=1)
        dense_layers = []

        for i in range(num_layers):
            units = hp.Int('units_{}'.format(i),
                           min_value=model_configuration.model_with_optimization.units_range.start,
                           max_value=model_configuration.model_with_optimization.units_range.end,
                           step=model_configuration.model_with_optimization.units_range.step,
                           default=10)
            activation = hp.Choice('activations_{}'.format(i),
                                   model_configuration.model_with_optimization.activations_functions,
                                   default='sigmoid')
            dense_layers.append(tensorflow.keras.layers.Dense(units=units, activation=activation, dtype='float64'))

        super().__init__(
            loss=self.loss,
            trainable_variables=trainable_variables,
            dense_list=dense_layers
        )

    def loss(self):
        self.__wrapper.set_current_model(self)
        return self.__basic_loss()


class ModelWithOptimizationWrapper:
    def __init__(self, loss, trainable_variables: TrainableVariables):
        self.__loss = loss
        self.__trainable_variables = trainable_variables
        self.__model = None

        self.__tuner = MyRandomSearch(
            self.build_model,
            objective='loss',
            max_trials=model_configuration.model_with_optimization.number_of_trials,
            directory='pinn_tuning',
            project_name='eq',
            overwrite=True
        )

    def init(self, inputs):
        self.__tuner.search_space_summary()
        self.__tuner.search(
            x=inputs,
            y=None,
            epochs=model_configuration.model_with_optimization.epochs,
            batch_size=100,
            verbose=1
        )

        best_hps = self.__tuner.get_best_hyperparameters(num_trials=1)[0]

        num_layers = best_hps.get('num_layers')
        print("=== Najlepsza konfiguracja modelu ===")
        print(f"- Learning rate: {best_hps.get('learning_rate'):.6f}")
        print(f"- Liczba warstw: {num_layers}")
        print("--------------------------------------")

        for i in range(num_layers):
            units = best_hps.get(f'units_{i}')
            act = best_hps.get(f'activations_{i}')
            print(f"Warstwa {i}:")
            print(f"  > Liczba neuronów: {units}")
            print(f"  > Funkcja aktywacji: {act}")

        self.__model = self.__tuner.hypermodel.build(best_hps)

        self.__model(inputs)

    def set_current_model(self, model):
        self.__model = model

    def __call__(self, inputs):
        return self.__model(inputs)

    def train_step(self, data=None):
        return self.__model.train_step(data=data)

    def get_gradients(self):
        return self.__model.get_gradients()

    def build_model(self, hp):
        model = ModelWithOptimization(
            wrapper=self,
            loss=self.__loss,
            trainable_variables=self.__trainable_variables,
            hp=hp
        )

        lr = hp.Float('learning_rate', min_value=5e-3, max_value=0.1, sampling='log', default=0.1)
        model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=lr))
        return model
