# TensorFlow Image Classifier

This project is a TensorFlow-based image classification pipeline. It allows users to train a deep learning model on a dataset of images, evaluate its performance, and make predictions on single images.

## Features
- Load and preprocess image datasets.
- Train a custom image classification model.
- Evaluate the model's performance with metrics like precision, recall, and F1-score.
- Save trained models for future use.
- Predict the class of a single image.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tf-image-classifier.git
   cd tf-image-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
Run the `main.py` script to train the model:
```bash
python main.py --data_dir /path/to/dataset --img_height 224 --img_width 224 --batch_size 32 --epochs 15 --save_dir models --model_name my_model
```

#### Arguments:
- `--data_dir`: Path to the directory containing image classes (required).
- `--img_height`: Height of the input images (default: 224).
- `--img_width`: Width of the input images (default: 224).
- `--batch_size`: Batch size for training (default: 32).
- `--epochs`: Number of training epochs (default: 15).
- `--save_dir`: Directory to save the trained model (default: `models`).
- `--model_name`: Name of the model (default: `model`).

### Example
Assume your dataset is structured as follows:
```
/path/to/dataset/
    class1/
        img1.jpg
        img2.jpg
    class2/
        img3.jpg
        img4.jpg
```

Run the training script:
```bash
python main.py --data_dir /path/to/dataset --epochs 10
```

### Evaluating the Model
After training, the script will evaluate the model on the validation dataset and print metrics like precision, recall, F1-score, and overall accuracy.

### Predicting a Single Image
To predict a single image, you can use the `predict_single_image` function from `src.utils`. Example:
```python
from src.utils import predict_single_image

image_path = "/path/to/image.jpg"
model_path = "/path/to/saved_model"
class_names = ["class1", "class2"]

prediction = predict_single_image(image_path, model_path, class_names)
print(f"Predicted class: {prediction}")
```

## Requirements
- Python 3.7+
- TensorFlow 2.x
- dependencies listed in `requirements.txt`

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.