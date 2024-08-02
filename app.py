
from flask import Flask, render_template , request, jsonify
from PIL import  Image
from model.model import load_model, preprocess_image
import tensorflow as tf
import numpy as np


app = Flask(__name__)


model =  load_model()

@app.route('/')
def homepage():  # put application's code here
    return render_template('homepage.html')



@app.route('/predict', methods=['POST'])
def predictImage():
    # check if the file exists or not
    if 'file' not in request.files:
        return jsonify({"Error!" : "No image received. Please try again! "}), 400


    # Get the image file from the request
    image_file = request.files['file']
    # Open the image file
    image = Image.open(image_file.stream)
    # Preprocess the image
    input_data = preprocess_image(image)

    # Convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data)

    # Make predictions using the model
    predictions = model(input_tensor)

    class_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
    # Assuming your model outputs probabilities for classes, get the predicted class
    predicted_class = class_names[np.argmax(predictions.numpy(), axis=1)[0]]
    print("The predicted class is ", predicted_class)
    # Return the predicted class as JSON
    return jsonify(predicted_class = predicted_class)


if __name__ == '__main__':
    print("Starting the server")
    app.run()
