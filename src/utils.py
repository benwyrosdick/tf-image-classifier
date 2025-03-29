import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

def evaluate_model(model, test_ds, class_names, save_dir=None):
    """
    Evaluate the model on test data and generate performance metrics.
    
    Args:
        model: Trained TensorFlow model
        test_ds: Test dataset
        class_names: List of class names
        save_dir: Directory to save evaluation results
    
    Returns:
        results: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_true = []
    y_pred = []
    
    # Reset the test dataset
    test_ds.reset()
    
    # Get true labels and predictions
    for i in range(len(test_ds)):
        images, labels = test_ds[i]
        predictions = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        
        if i >= len(test_ds) - 1:
            break
    
    # Calculate evaluation metrics
    labels = list(range(len(class_names)))  # Ensure labels match class indices
    report = classification_report(y_true, y_pred, target_names=class_names, labels=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot confusion matrix
    if save_dir:
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    
    # Return evaluation results
    return {
        'classification_report': report,
        'confusion_matrix': cm
    }

def predict_single_image(model, image_path, class_names, img_height=224, img_width=224):
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained TensorFlow model
        image_path: Path to the image file
        class_names: List of class names
        img_height: Target image height
        img_width: Target image width
    
    Returns:
        predicted_class: Predicted class name
        confidence: Prediction confidence
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    return predicted_class, confidence