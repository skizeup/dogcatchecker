import tensorflow as tf
   from tensorflow.keras.preprocessing import image
   import numpy as np

   def predict_image(model, img_path):
       img = image.load_img(img_path, target_size=(224, 224))
       img_array = image.img_to_array(img)
       img_array = np.expand_dims(img_array, 0)
       img_array /= 255.

       prediction = model.predict(img_array)
       if prediction[0][0] > prediction[0][1]:
           return f"C'est probablement un chat (confiance: {prediction[0][0]:.2f})"
       else:
           return f"C'est probablement un chien (confiance: {prediction[0][1]:.2f})"

   # Charger le modèle entraîné
   model = tf.keras.models.load_model('dog_cat_classifier.h5')

   # Liste des images à tester
   image_paths = [
       '/Users/lucas/Downloads/chien.jpg',
       '/Users/lucas/Downloads9998.jpg',
       '/Users/lucas/Downloads/chat.jpg'
   ]

   # Tester le modèle sur chaque image
   for path in image_paths:
       result = predict_image(model, path)
       print(f"Image: {path}")
       print(f"Résultat: {result}\n")
