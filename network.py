import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(x, y):
    # Input tensor: shape (13, 26, 7)
    input_tensor = layers.Input(shape=(13, 26, 7))

    # Zero-padding to make the input tensor shape (30, 30, 7)
    padded_input = layers.ZeroPadding2D(padding=((8, 9), (2, 2)))(input_tensor)

    # First convolutional layer: 32 filters, 4x4 kernel, stride 2
    x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation='relu')(padded_input)

    # Second convolutional layer: 64 filters, 4x4 kernel, stride 2
    x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu')(x)

    # Third convolutional layer: 64 filters, 3x3 kernel, stride 1
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu')(x)

    # Flatten the output from the convolutional layers
    x = layers.Flatten()(x)

    # Fully connected layer with 128 units
    x = layers.Dense(units=128, activation='relu')(x)

    # Output layer with 1 unit (Q-value output)
    output = layers.Dense(units=1)(x)

    # Build the model
    model = models.Model(inputs=input_tensor, outputs=output)

    return model

# # Create the model
# model = build_model()

# # Print the model summary
# model.summary()
