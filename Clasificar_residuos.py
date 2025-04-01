import tensorflow as tf
import numpy as np

from src.Definir_modelo import model
from src.Preproces_Datos import img_height, img_width

# Función para predecir la clase de una imagen
def classify_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Clasificar una nueva imagen
image_path = 'path_to_new_image.jpg'
predicted_class = classify_image(image_path)
print('Predicted class:', predicted_class)
