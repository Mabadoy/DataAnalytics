import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model_path = "/content/CIFAR_10_tens.h5"  # Ensure the file is in your working directory
model = load_model(model_path)

# Define CIFAR-10 labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load a new image from a different dataset
image_path = "/content/re-test-fig.jpeg"  # Replace with your test image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
img_resized = cv2.resize(img_rgb, (32, 32))  # Resize to model input size

# Normalize the image
img_array = img_resized.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predicted_class = model.predict(img_array).argmax()
predicted_label = labels[predicted_class]


# Display the image and prediction
plt.imshow(img_rgb)
plt.axis("off")
plt.title(f"Predicted: {predicted_label}")
plt.show()
print(f"Predicted label: {predicted_label}")