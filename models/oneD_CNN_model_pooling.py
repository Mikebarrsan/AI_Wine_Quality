import tensorflow as tf

def create_model(
    num_classes: int,
    num_attributes: int = 11,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:

    model = tf.keras.Sequential(name="wine_quality_1d_cnn_v2")

    model.add(tf.keras.layers.Input(shape=(num_attributes, 1), name="wine_features"))

    model.add(tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1d_64_k3"
    ))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_conv_64"))

    model.add(tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1d_64_k5"
    ))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_conv_64_k5"))

    model.add(tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool"))

    model.add(tf.keras.layers.Dense(128, activation="relu", name="dense_128"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_128"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_128"))

    model.add(tf.keras.layers.Dense(64, activation="relu", name="dense_64"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_64"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_64"))

    model.add(tf.keras.layers.Dense(32, activation="relu", name="dense_32"))
    model.add(tf.keras.layers.Dropout(dropout_rate / 2, name="dropout_32"))

    model.add(tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="quality_output"
    ))

    return model