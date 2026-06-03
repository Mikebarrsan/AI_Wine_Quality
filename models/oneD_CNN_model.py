"""
1D-CNN model for wine quality classification.

Architecture based on the paper:
Prediction of Red Wine Quality Using One-Dimensional Convolutional Neural Networks.

Pipeline represented by the model:
Input wine features -> 1D Convolution -> Flatten -> 4 Dense layers -> Softmax output.
The model also includes Dropout and Batch Normalization blocks to reduce overfitting.
"""

import tensorflow as tf


def create_model(
    num_classes: int,
    num_attributes: int = 11,
    dropout_rate: float = 0.30,
) -> tf.keras.Model:
    """
    Create the 1D-CNN model used for wine quality classification.

    Parameters
    ----------
    num_classes : int
        Number of wine quality classes after label encoding.
    num_attributes : int
        Number of input wine attributes/features.
    dropout_rate : float
        Dropout rate used after the first three dense layers.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model.
    """

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    if num_attributes < 1:
        raise ValueError("num_attributes must be at least 1.")

    model = tf.keras.Sequential(name="wine_quality_1d_cnn")

    # Input - Wine attributes
    model.add(tf.keras.layers.Input(shape=(num_attributes, 1), name="wine_features"))

    # 1D-CNN block: captures local relationships between neighboring attributes.
    model.add(tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d_block"
    ))

    # The CNN output is flattened before entering the dense layers.
    model.add(tf.keras.layers.Flatten(name="flatten"))

    # Dense block: 4 fully connected layers.
    # The first three use BatchNorm + Dropout, matching the paper's idea
    # of using these blocks to improve robustness and reduce overfitting.

    # The paper specifies that the flattened convolutional output is passed through 
    # four dense layers before the softmax output. Since the exact number of neurons 
    # is not provided, a progressively decreasing structure was used to reduce the 
    # representation size while preserving enough capacity for classification

    # Dense layer
    model.add(tf.keras.layers.Dense(128, name="dense_128"))
    # Batch Normalization and Dropout after the first three dense layers to improve generalization.
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_128"))
    model.add(tf.keras.layers.Activation("relu", name="relu_128"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_128"))

    # Dense layer
    model.add(tf.keras.layers.Dense(64, name="dense_64"))
    # Batch Normalization and Dropout
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_64"))
    model.add(tf.keras.layers.Activation("relu", name="relu_64"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_64"))

    # Dense layer
    model.add(tf.keras.layers.Dense(32, name="dense_32"))
    # Batch Normalization and Dropout
    model.add(tf.keras.layers.BatchNormalization(name="batch_norm_32"))
    model.add(tf.keras.layers.Activation("relu", name="relu_32"))
    model.add(tf.keras.layers.Dropout(dropout_rate, name="dropout_32"))

    # Fourth and finaldense layer before softmax
    model.add(tf.keras.layers.Dense(16, activation="relu", name="dense_16"))

    # Softmax output layer for multi-class classification.
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="quality_output"))

    return model
