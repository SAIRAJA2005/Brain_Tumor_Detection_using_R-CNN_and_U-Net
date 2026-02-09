from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# =============================
# Dice Coefficient
# =============================

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# =============================
# U-Net Model (channels_last)
# =============================

def unet_model(input_size=(120, 120, 1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, 3, activation="relu", padding="same")(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, 3, activation="relu", padding="same")(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, 3, activation="relu", padding="same")(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, 3, activation="relu", padding="same")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, 3, activation="relu", padding="same")(c4)
    c4 = BatchNormalization()(c4)

    # Decoder
    u5 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation="relu", padding="same")(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, 3, activation="relu", padding="same")(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, 3, activation="relu", padding="same")(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, 3, activation="relu", padding="same")(c7)
    c7 = BatchNormalization()(c7)

    outputs = Conv2D(1, 1, activation="sigmoid")(c7)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=[dice_coef],
    )

    return model
