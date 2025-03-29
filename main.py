import os
import argparse
from datetime import datetime
import tensorflow as tf
from src.data_loader import load_and_preprocess_data
from src.model import create_model
from src.train import train_model
from src.utils import evaluate_model, predict_single_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an image classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing image classes')
    parser.add_argument('--img_height', type=int, default=224, help='Image height')
    parser.add_argument('--img_width', type=int, default=224, help='Image width')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    
    args = parser.parse_args()
    
    # Check if GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_ds, val_ds, class_names = load_and_preprocess_data(
        args.data_dir,
        args.img_height,
        args.img_width,
        args.batch_size
    )
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create the model
    print("Creating model...")
    input_shape = (args.img_height, args.img_width, 3)  # RGB images
    model = create_model(input_shape, len(class_names))
    model.summary()

    # Compile the model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(args.save_dir, f"{args.model_name}_{timestamp}")
    
    # Train the model
    print("Training model...")
    trained_model, history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        save_dir=model_path,
    )
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = evaluate_model(
        trained_model,
        val_ds,
        class_names,
        save_dir=model_path,
    )
    
    # Print classification report
    print("\nClassification Report:")
    for class_name in class_names:
        precision = eval_results['classification_report'][class_name]['precision']
        recall = eval_results['classification_report'][class_name]['recall']
        f1_score = eval_results['classification_report'][class_name]['f1-score']
        print(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}")
    
    print("\nOverall Accuracy:", eval_results['classification_report']['accuracy'])
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()