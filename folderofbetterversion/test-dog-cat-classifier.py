import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array /= 255.  # Normalize the image

        prediction = model.predict(img_array)
        if prediction[0][0] > prediction[0][1]:
            return f"C'est probablement un chat (confiance: {prediction[0][0]:.2f})"
        else:
            return f"C'est probablement un chien (confiance: {prediction[0][1]:.2f})"
    except Exception as e:
        return f"Erreur lors du traitement de l'image {img_path}: {str(e)}"

# Charger le modèle entraîné
model = tf.keras.models.load_model('dog_cat_classifier.h5')

# Liste des images à tester
image_paths = [
    '/Users/lucas/Downloads/chien.jpg',
    '/Users/lucas/Downloads/chat.jpg',
    '/Users/lucas/Downloads/9998.jpg'  # Make sure this path is correct
]

# Tester le modèle sur chaque image
for path in image_paths:
    result = predict_image(model, path)
    print(f"Image: {path}")
    print(f"Résultat: {result}\n")
