import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, img_height=224, img_width=224, batch_size=32, validation_split=0.2):
    """
    Load and preprocess images from directory structure.
    
    Args:
        data_dir: Directory containing class subdirectories of images
        img_height: Target image height
        img_width: Target image width
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
        class_names: List of class names
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Load training data
    train_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    val_ds = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class names from directory
    class_names = list(train_ds.class_indices.keys())
    
    return train_ds, val_ds, class_names