import tensorflow as tf

def create_model(
    num_classes: int,
    num_attributes: int = 11,
    dropout_rate: float = 0.40,
    filters: int = 32,
    kernel_size: int = 5,
) -> tf.keras.Model:

    model = tf.keras.Sequential(name="wine_quality_1d_cnn_generalized")

    regularizer = tf.keras.regularizers.l2(0.003)

    model.add(tf.keras.layers.Input(
        shape=(num_attributes, 1),
        name="wine_features"
    ))

    model.add(tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
        name="conv1d_32_k5"
    ))

    model.add(tf.keras.layers.SpatialDropout1D(
        0.20,
        name="spatial_dropout_conv"
    ))

    model.add(tf.keras.layers.BatchNormalization(
        name="batch_norm_conv"
    ))

    model.add(tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="max_pool_1d"
    ))

    model.add(tf.keras.layers.GlobalAveragePooling1D(
        name="global_avg_pool"
    ))

    model.add(tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_32"
    ))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_32"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_32"))

    model.add(tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_16"
    ))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_16"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_16"))

    model.add(tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_8"
    ))
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_8"))
    model.add(tf.keras.layers.Dropout(dropout_rate / 2, name="dropout_8"))

    model.add(tf.keras.layers.Dense(
        4,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_4"
    ))

    model.add(tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="quality_output"
    ))

    return model