import copy
import tensorflow
from objects.TrainableVariables import TrainableVariables


class WangModel(tensorflow.keras.Model):
    # Zamiast 'dense_list' przekazujemy parametry definiujące rozmiar sieci
    def __init__(self, loss, trainable_variables: TrainableVariables, layers_Z, hidden_dim=10,
                 activation: str = "sigmoid", optimizer=None):
        super(WangModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.activation = activation

        # 1. Enkodery U i V przekształcające wejście w wysokowymiarową przestrzeń
        self.encoder_U = tensorflow.keras.layers.Dense(units=hidden_dim, activation=activation, dtype='float64')
        self.encoder_V = tensorflow.keras.layers.Dense(units=hidden_dim, activation=activation, dtype='float64')

        # 2. Inicjalizacja pierwszego stanu ukrytego H^(1)
        self.layer_H1 = tensorflow.keras.layers.Dense(units=hidden_dim, activation=activation, dtype='float64')

        # 3. Lista warstw wyliczających suwaki (bramki) Z dla każdej warstwy ukrytej
        self.layers_Z = layers_Z

        # 4. Warstwa wyjściowa
        self.out_dense = tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')

        self._loss = loss
        self._custom_trainable_variables = trainable_variables
        self.optimizer = optimizer
        self.__grads = None

    def call(self, inputs):
        # KROK 1: Wyliczenie "ściągawek" U i V na podstawie danych wejściowych
        U = self.encoder_U(inputs)
        V = self.encoder_V(inputs)

        # KROK 2: Obliczenie pierwszego stanu ukrytego H
        H = self.layer_H1(inputs)

        # KROK 3: Pętla przez kolejne warstwy - punktowe mnożenie z U i V
        for layer_z in self.layers_Z:
            Z = layer_z(H)
            # Operator '*' w TensorFlow działa jako mnożenie punktowe (element-wise)
            H = (1.0 - Z) * U + Z * V

        # KROK 4: Obliczenie predykcji końcowej
        output = self.out_dense(H)
        return output

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

        # UWAGA: W nowej architekturze self.trainable_variables ma inny układ.
        # Zamiast polegać na indeksach [-2:], bezpieczniej jest pobrać wagi bezpośrednio z warstwy wyjściowej:
        last_layer_weights = self.out_dense.trainable_variables

        grad_data = tape.gradient(tensorflow.convert_to_tensor(conditions_data, dtype=tensorflow.float64),
                                  last_layer_weights)
        grad_pde = tape.gradient(tensorflow.convert_to_tensor(loss_pde, dtype=tensorflow.float64), last_layer_weights)
        grad_bc = tape.gradient(tensorflow.convert_to_tensor(conditions, dtype=tensorflow.float64), last_layer_weights)

        del tape

        self.__grads = {
            'grad_data': grad_data,
            'grad_pde': grad_pde,
            'grad_bc': grad_bc,
        }
        return {
            'loss': current_loss
        }

    def get_gradients(self):
        return self.__grads

    def __deepcopy__(self, memo):
        # Ponieważ zrezygnowaliśmy z dynamicznego Sequential, deepcopy staje się znacznie prostsze.
        # Wystarczy odtworzyć model z zapisanymi parametrami wymiarowości i głębokości.
        new_model = WangModel(
            loss=copy.deepcopy(self._loss, memo),
            trainable_variables=copy.deepcopy(self._custom_trainable_variables, memo),
            hidden_dim=self.hidden_dim,
            layers_Z=copy.deepcopy(self.layers_Z, memo),
            activation=self.activation,
            optimizer=copy.deepcopy(self.optimizer, memo)
        )

        memo[id(self)] = new_model
        return new_model