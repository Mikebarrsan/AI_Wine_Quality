import tensorflow as tf

def create_model(num_classes):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            16,
            activation='relu',
            input_shape=(11,)
        ),

        tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )
    ])

    return model