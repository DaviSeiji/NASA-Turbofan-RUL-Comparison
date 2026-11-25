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


# Constrói a TCN com convoluções causais e blocos residuais.
def build_tcn(input_shape, params):

    blocks = params['num_blocks']
    kernel = params['kernel_size']

    if blocks < 1 or kernel < 2:
        raise ValueError('A TCN precisa de blocos positivos e kernel de pelo menos 2.')

    # Cada bloco tem duas convoluções, com dilatações 1, 2, 4, 8...
    receptive_field = 1 + 2 * (kernel - 1) * (2**blocks - 1)

    if receptive_field < input_shape[0]:
        raise ValueError('O campo receptivo da TCN precisa cobrir toda a janela.')

    inputs = keras.layers.Input(shape=input_shape)
    x = inputs

    for index in range(blocks):
        residual = x

        for _ in range(2):
            x = keras.layers.Conv1D(
                params['filters'], kernel, padding='causal', dilation_rate=2**index,
                activation='relu',
            )(x)
            x = keras.layers.SpatialDropout1D(params['dropout_rate'])(x)

        if residual.shape[-1] != params['filters']:
            residual = keras.layers.Conv1D(params['filters'], 1)(residual)

        x = keras.layers.Add()([residual, x])
        x = keras.layers.Activation('relu')(x)

    # Usa o último ciclo, que pode receber informação de toda a janela.
    x = keras.layers.Cropping1D((input_shape[0] - 1, 0))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name='tcn_rul')
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


# Constrói o iTransformer adaptado para prever um único RUL.
def build_itransformer(input_shape, params):

    inputs = keras.layers.Input(shape=input_shape)

    # Cada variável vira um token com os 30 valores da sua janela.
    x = keras.layers.Permute((2, 1), name='variable_tokens')(inputs)
    x = keras.layers.Dense(params['d_model'], name='variable_projection')(x)
    x = keras.layers.Dropout(params['dropout_rate'])(x)

    # A atenção relaciona variáveis, sem embedding de posição temporal.
    for _ in range(params['num_layers']):
        x = transformer_encoder(
            x, params['d_model'], params['num_heads'], params['ff_dim'],
            params['dropout_rate'],
        )

    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(params['dropout_rate'])(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name='itransformer_rul')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse', metrics=['mae'],
    )

    return model
