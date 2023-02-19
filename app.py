import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify

host = "0.0.0.0"

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict_image():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return 'No file uploaded'
    file = request.files['file']

    # Save the image to a temporary file
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Load and preprocess the image
    height = 180
    width = 180
    channels = 3
    img = load_img(file_path, target_size=(height, width))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.reshape(img_array, [1, height, width, channels])

    # Load the model and make a prediction
    model = tf.keras.models.load_model('nova.h5')
    prediction = model.predict(img_array)    
    # Delete the temporary file
    os.remove(file_path)

    # Return the prediction as JSON
    return jsonify({'prediction': 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'})

if __name__ == '__main__':
    app.run(host=host, debug=False)
