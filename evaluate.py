import argparse
import tensorflow as tf
from data.preprocess import load_data
from models.vit import VisionTransformer
from models.swin import SwinTransformer
from models.maxvit import MaxViT
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix

# Define model choices
MODEL_CLASSES = {
    "vit": VisionTransformer,
    "swin": SwinTransformer,
    "maxvit": MaxViT
}

def evaluate_model(model_name, model_path, data_dir, batch_size=32):
    """Evaluates a trained Vision Transformer model."""
    
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(MODEL_CLASSES.keys())}")

    # Load dataset
    _, val_data = load_data(data_dir, batch_size=batch_size)

    # Load trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from '{model_path}'")

    # Evaluate model
    results = model.evaluate(val_data)
    print(f"Evaluation Results - Loss: {results[0]}, Accuracy: {results[1]}")

    # Compute predictions
    y_true = val_data.classes
    y_pred = model.predict(val_data)

    # Calculate metrics
    calculate_metrics(y_true, y_pred, model_name)

    # Plot confusion matrix
    class_names = list(val_data.class_indices.keys())  # Extract class labels
    plot_confusion_matrix(y_true, y_pred, class_names, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Vision Transformer model.")
    parser.add_argument("--model", type=str, choices=MODEL_CLASSES.keys(), required=True, help="Model to evaluate (vit, swin, maxvit)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )