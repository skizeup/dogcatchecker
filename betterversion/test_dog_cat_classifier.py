import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os

# Fonctions utilitaires
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def train_model(train_dir, validation_dir, epochs=10):
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    return model

def predict_image(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.

    prediction = model.predict(img_array)
    if prediction[0][0] > prediction[0][1]:
        return f"C'est probablement un chat (confiance: {prediction[0][0]:.2f})"
    else:
        return f"C'est probablement un chien (confiance: {prediction[0][1]:.2f})"

# Exemple d'utilisation
train_dir = '/Users/lucas/Desktop/betterversion/Data/train/'
validation_dir = '/Users/lucas/Desktop/betterversion/Data/validation'

# Entraînement du modèle
model = train_model(train_dir, validation_dir)
model.save('dog_cat_classifier.h5')

# Chargement d'un modèle entraîné
# model = tf.keras.models.load_model('dog_cat_classifier.h5')

# Test sur quelques images
#image_paths = ['chemin/image1.jpg', 'chemin/image2.jpg', 'chemin/image3.jpg']
#for path in image_paths:
#    result = predict_image(model, path)
#    print(f"Image: {path}")
#    print(f"Résultat: {result}\n")
