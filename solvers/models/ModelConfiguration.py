import tensorflow


class Range:
    def __init__(self, start, end, step=None):
        if start >= end:
            raise "start must be less than end"
        self.start = start
        self.end = end
        self.step = step


class ModelWithOptimizationConfiguration:
    def __init__(self, number_of_layers_range: Range = None, units_range: Range = None,
                 activations_functions: [str] = None, number_of_trials: int = 10, epochs:int = 500):
        self.number_of_layers_range = number_of_layers_range if number_of_layers_range is not None else Range(
            start=1,
            end=5,
        )
        self.units_range = units_range if units_range is not None else Range(
            start=1,
            end=100,
            step=10
        )
        self.activations_functions = activations_functions if activations_functions is not None \
            else ['tanh', 'sigmoid', 'swish']
        self.number_of_trials = number_of_trials
        self.epochs = epochs


class ModelConfiguration:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(ModelConfiguration, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.dense_list = [
                tensorflow.keras.layers.Dense(units=10, activation='sigmoid', dtype='float64')
            ]
            self.model_with_optimization = None
            self._initialized = True

    def configure(self, dense_list=None, model_with_optimization=None):
        self.dense_list = dense_list if dense_list is not None else self.dense_list
        self.model_with_optimization = model_with_optimization if model_with_optimization is not None \
            else self.model_with_optimization

    def can_optimize(self):
        return self.model_with_optimization is not None

    def get_optimizer(self, learning_rate: float):
        return tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
