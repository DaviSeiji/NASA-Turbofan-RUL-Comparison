import keras

#Essa classe serve para adicionar embeddings de posição à entrada do Transformer. 
#Ela cria um vetor treinável para cada posição temporal, permitindo que o modelo capture a ordem dos dados de entrada.
#Assim podemos saber de qual ciclo cada ponto de dados se refere, o que é crucial para tarefas de previsão de séries temporais.
@keras.utils.register_keras_serializable(package="RUL")
class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, window_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size)
        self.d_model = int(d_model)

    #Cria um vetor treinável para cada posição temporal.
    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.window_size, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    #Adiciona a informação de posição à entrada.
    def call(self, inputs):
        return inputs + self.position_embeddings

    #Guarda os parâmetros necessários para reconstruir a camada.
    def get_config(self):
        return {
            **super().get_config(),
            "window_size": self.window_size,
            "d_model": self.d_model,
        }
