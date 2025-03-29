import os
import tensorflow as tf
import matplotlib.pyplot as plt

def train_model(model, train_ds, val_ds, save_dir, epochs=10):
    """
    Train the model and save the results.
    
    Args:
        model: Compiled TensorFlow model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs
        save_dir: Directory to save the model
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Create callbacks
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "checkpoint.keras"),  # Added .keras extension
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(os.path.join(save_dir, "final_model.keras"))  # Added .keras extension
    
    # Save the model in TFLite format for use in various environments
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(save_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)
    
    # Save the model in SavedModel format for TensorFlow Serving
    tf.saved_model.save(model, os.path.join(save_dir, "saved_model"))
    
    # Plot and save training history
    plot_training_history(history, save_dir)
    
    print(f"Model saved to {save_dir}")
    
    return model, history

def plot_training_history(history, save_dir):
    """
    Plot and save training history graphs.
    
    Args:
        history: Training history
        save_dir: Directory to save the plots
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()