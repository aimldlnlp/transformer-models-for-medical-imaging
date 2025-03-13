import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data.preprocess import load_data
from models.vit import VisionTransformer
from models.swin import SwinTransformer
from models.maxvit import MaxViT
from utils.visualization import plot_training_history

# Define model choices
MODEL_CLASSES = {
    "vit": VisionTransformer,
    "swin": SwinTransformer,
    "maxvit": MaxViT
}

def train_model(model_name, data_dir, epochs=10, batch_size=32, learning_rate=1e-4):
    """Trains the specified Vision Transformer model."""
    
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(MODEL_CLASSES.keys())}")

    # Load dataset
    train_data, val_data = load_data(data_dir, batch_size=batch_size)

    # Initialize model
    model_class = MODEL_CLASSES[model_name]
    model_instance = model_class(learning_rate=learning_rate)
    model = model_instance.model  # Access compiled model

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save trained model
    model.save(f"saved_models/{model_name}_kidney_stone.h5")
    print(f"Model saved as 'saved_models/{model_name}_kidney_stone.h5'")

    # Plot training results
    plot_training_history(history, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model.")
    parser.add_argument("--model", type=str, choices=MODEL_CLASSES.keys(), required=True, help="Model to train (vit, swin, maxvit)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )