import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the model from the .h5 file
model = load_model("colon1.h5")

# Load and preprocess the image
img_path = 'train.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the pixel values

# Make predictions
predictions = model.predict(img_array)

# Interpret the predictions
class_indices = {
    0: "Normal",
    1: "Ulcerative colitis",
    2: "Polyp",
    3: "Esophagitis"
}

predicted_class_index = np.argmax(predictions)
predicted_class_label = class_indices[predicted_class_index]

# Display the image and predicted class
import matplotlib.pyplot as plt

plt.imshow(img)
plt.title("Predicted class: " + predicted_class_label)
plt.show()
