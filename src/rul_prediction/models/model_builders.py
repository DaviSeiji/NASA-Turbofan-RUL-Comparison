import keras
from src.rul_prediction.models.transformer_layers import PositionalEmbedding

#Constrói e compila a CNN-LSTM com os parâmetros recebidos.
def build_cnn_lstm(input_shape, params):

    cnn_layers = params.get('cnn_layers', 1)
    lstm_layers = params.get('lstm_layers', 1)

    for name, count in (('cnn_layers', cnn_layers), ('lstm_layers', lstm_layers)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f'{name} deve ser um inteiro positivo.')
        
    if (input_shape[0] is not None and input_shape[0] - cnn_layers * (params['kernel_size'] - 1) < 1):
        raise ValueError('Camadas CNN e kernel_size excedem o comprimento da janela.')

    layers = [keras.layers.Input(shape=input_shape)]

    for _ in range(cnn_layers):
        layers.append(keras.layers.Conv1D(
            filters=params['filters'], kernel_size=params['kernel_size'], activation='relu',
        ))

    for index in range(lstm_layers):
        layers.append(keras.layers.LSTM(
            units=params['lstm_units'], return_sequences=index < lstm_layers - 1,
        ))

    layers.extend([
        keras.layers.Dropout(params['dropout_rate']),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1),
    ])

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse', metrics=['mae'],
    )
    return model


#Função que constroí um bloco de encoder do Transformer.
def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout):

    if d_model % num_heads != 0:
        raise ValueError('d_model deve ser divisível por num_heads.')

    #Cada bloco vai ter:
    #Camada de atenção Multi-Head, seguida de normalização e dropout.
    attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, output_shape=d_model, dropout=dropout)(inputs, inputs)
    attention = keras.layers.Dropout(dropout)(attention)

    #Adiciona a entrada residual e normaliza.
    x = keras.layers.Add()([inputs, attention])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    #Camada Feed-Forward, seguida de normalização e dropout.
    ff = keras.layers.Dense(ff_dim, activation='relu')(x)
    ff = keras.layers.Dropout(dropout)(ff)
    ff = keras.layers.Dense(d_model)(ff)
    ff = keras.layers.Dropout(dropout)(ff)
    x = keras.layers.Add()([x, ff])

    #Adiciona a entrada residual e normaliza novamente.
    return keras.layers.LayerNormalization(epsilon=1e-6)(x)


# Constrói e compila o Transformer.
def build_transformer(input_shape, params):

    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(params['d_model'])(inputs)
    x = PositionalEmbedding(input_shape[0], params['d_model'])(x)

    for _ in range(params['num_layers']):
        x = transformer_encoder(x, params['d_model'], params['num_heads'],params['ff_dim'], params['dropout_rate'])

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(params['dropout_rate'])(x)

    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name='transformer_rul')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mse', metrics=['mae'])
    
    return model
