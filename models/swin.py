import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_hub as hub

class SwinTransformer:
    """Swin Transformer model for image classification."""

    def __init__(self, input_shape=(224, 224, 3), num_classes=4, learning_rate=1e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Builds the Swin Transformer model architecture."""
        swin_layer = hub.KerasLayer(
            "https://tfhub.dev/sayakpaul/swin_transformer_tiny_patch4_window7_224/1", trainable=True
        )

        inputs = layers.Input(shape=self.input_shape, name="input_layer")
        x = swin_layer(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name="output_layer")(x)

        model = models.Model(inputs, outputs, name="Swin_Transformer")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def summary(self):
        """Prints the model summary."""
        self.model.summary()

if __name__ == "__main__":
    swin = SwinTransformer()
    swin.summary()