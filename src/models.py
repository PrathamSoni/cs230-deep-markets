import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

cnn1 = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((10, 512, 1)),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(10, 512, 1)),
    tf.keras.layers.Conv2D(4, (1, 1), activation='relu', input_shape=(10, 512, 1)),
    tf.keras.layers.Flatten(),
])


def get_linear(hidden_dim=0):
    if hidden_dim == 0:
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
        ])
    else:
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
        ])


lstm1 = tf.keras.Sequential([
    tf.keras.layers.Reshape((10, 12)),
    tf.keras.layers.LSTM(32),
])

lstm2 = tf.keras.Sequential([
    tf.keras.layers.LSTM(8),
])


def get_dual_model():
    input1 = tf.keras.layers.Input(shape=(10, 512))
    news = get_linear(2)(input1)
    news = tf.keras.Model(inputs=input1, outputs=news)

    input2 = tf.keras.layers.Input(shape=(10, 12, 1))
    sp = tf.keras.layers.Reshape((10, 12))(input2)
    sp = get_linear(64)(sp)
    sp = tf.keras.Model(inputs=input2, outputs=sp)

    features = tf.keras.layers.concatenate([sp.output, news.output])

    return tf.keras.Model(inputs=[news.input, sp.input], outputs=features)


def get_final(encoder, type_final):
    output = encoder.output
    if type_final == 0:
        pass
    if type_final == 1:
        output = tf.keras.layers.Dense(4, activation='relu')(output)
    if type_final == 2:
        output = tf.keras.layers.Dense(4, activation='relu')(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Dense(4, activation='relu')(output)
    if type_final == 3:
        output = tf.keras.layers.Dropout(.5)(output)
    if type_final == 4:
        output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    return tf.keras.Model(inputs=encoder.inputs, outputs=output)


def getModel(model_id=0, type_final=None):
    if model_id == 0:
        model = cnn1
    elif model_id == 1:
        model = get_linear(hidden_dim=0)
    elif model_id == 2:
        model = get_linear(hidden_dim=2)
    elif model_id == 3:
        model = lstm1
    elif model_id == 4:
        model = lstm2
    elif model_id == 5:
        encoder = get_dual_model()
        model = get_final(encoder, type_final)
        return model
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model
