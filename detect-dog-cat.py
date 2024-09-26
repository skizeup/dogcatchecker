import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import ssl


# Disable SSL verification (use with care) (better not touch i think lol)
ssl._create_default_https_context = ssl._create_unverified_context

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def predict_dog_or_cat(img_path, confidence_threshold=0.1):
    # Load image and preprocess it
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make the prediction
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=1000)[0]  # Considere all classe

    # Check all predictions above the confidence threshold
    for _, label, score in decoded_preds:
        if score >= confidence_threshold:
            if 'dog' in label:
                return f"C'est probablement un dog (confiance: {score:.2f})"
            elif 'cat' in label:
                return f"C'est probablement un cat (confiance: {score:.2f})"

    return "Neither a dog nor a cat was detected with confidence."

# Change path of image and execute python3 detect-dog-cat.py in terminal
image_path = '/Users/lucas/Downloads/chien.jpg'
result = predict_dog_or_cat(image_path)
print(result)

