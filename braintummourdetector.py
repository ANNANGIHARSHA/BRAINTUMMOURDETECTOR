import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Set  paths to the dataset
image_path = 'path_to_your_image.jpg'  # Path to the MRI image

# Load  ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')

# Load an image from the specified path, target_size should match ResNet50's input size (224x224)
img = load_img(image_path, target_size=(224, 224))

# Convert the image to an array and preprocess it for the ResNet50 model
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess the image for ResNet50

# Predict the class probabilities for the input image
predictions = model.predict(img_array)

# Decode the predictions to human-readable class labels
decoded_predictions = decode_predictions(predictions, top=3)[0]

# Output the top 3 predictions
print('Predictions:')
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

# Based on your model, you would add a mapping of brain tumor classifications here
# e.g., a label such as "Tumor" if the model detects the presence of tumor cells.
