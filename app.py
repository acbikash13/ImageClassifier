
from flask import Flask, render_template , request, jsonify
from PIL import  Image
import tensorflow as tf
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)


def load_model():
     #saved model path
    print("Entering in the load model")
    print("Current directory is " +  os.getcwd())
    model_path = os.path.join(os.getcwd(), 'saved_model.h5')
    print("Directory contents:", os.listdir(current_dir))
    model = tf.keras.models.load_model(model_path,custom_objects=None , safe_mode = False)
    print("model is ")
    return model

def preprocess_image(image):
    # resize the image
    image_width = 264
    image_heigth = 264
    image =  image.resize((image_heigth,image_width))
    #convert the image into numpy array
    image_array = np.array(image)

    #Normalize the image array
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

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
