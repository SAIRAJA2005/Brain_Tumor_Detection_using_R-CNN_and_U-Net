import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf

# =============================
# Environment & TensorFlow setup
# =============================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

tf.keras.backend.set_image_data_format("channels_last")

# =============================
# Project imports
# =============================

from model import unet_model

# =============================
# Configuration
# =============================

img_size = 120
num_epoch = 30          # 🔥 increased
batch_size = 16

data_root = r"."
weights_root = r"E:/Final_Year_Project/BraTS_small/weights"
os.makedirs(weights_root, exist_ok=True)

# =============================
# Load dataset
# =============================

print("Loading dataset...")

train_X = np.load(os.path.join(data_root, f"x_{img_size}.npy"))
train_Y = np.load(os.path.join(data_root, f"y_{img_size}.npy"))

print("X min/max:", train_X.min(), train_X.max())
print("Y unique values:", np.unique(train_Y))

print("Dataset loaded")
print("Original X shape:", train_X.shape)
print("Original Y shape:", train_Y.shape)

# =============================
# Convert channels_first → channels_last
# =============================

if train_X.shape[1] == 1:
    train_X = np.transpose(train_X, (0, 2, 3, 1))
    train_Y = np.transpose(train_Y, (0, 2, 3, 1))

print("Converted X shape:", train_X.shape)
print("Converted Y shape:", train_Y.shape)

# =============================
# Safety checks
# =============================

assert train_X.shape == train_Y.shape
assert train_X.ndim == 4
assert train_X.shape[-1] == 1

# =============================
# Normalize data (IMPORTANT)
# =============================

train_X = train_X.astype("float32")
train_X /= np.max(train_X)

train_Y = train_Y.astype("float32")

# =============================
# Load model
# =============================

print("Loading the model...")
model = unet_model(input_size=(img_size, img_size, 1))
model.summary()
print("The model loaded")

# =============================
# Train the model
# =============================

print("Starting training...")

history = model.fit(
    train_X,
    train_Y,
    validation_split=0.25,
    batch_size=batch_size,
    epochs=num_epoch,
    shuffle=True,
    verbose=1,
)

# =============================
# Plot Dice coefficient
# =============================

if "dice_coef" in history.history:
    plt.figure()
    plt.plot(history.history["dice_coef"], label="Train Dice")
    plt.plot(history.history["val_dice_coef"], label="Val Dice")
    plt.title("Model Dice Coefficient")
    plt.ylabel("Dice Coef")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================
# Plot loss
# =============================

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True)
plt.show()

# =============================
# Save model weights
# =============================

weights_path = os.path.join(
    weights_root, f"dice_weights_{img_size}_{num_epoch}.h5"
)
model.save_weights(weights_path)

print("Model weights saved to:", weights_path)
