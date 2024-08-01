
import numpy as np
import tensorflow as tf
import os


def load_model():
    #saved model path
    print("Entering in the load model")
    model_path = './model/'
    print("Files in model directory:", os.listdir(model_path))
    model = tf.keras.models.load_model('./model/saved_model.h5',custom_objects=None , safe_mode = False)

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


import numpy as np
import tensorflow as tf
import os


def load_model():
    #saved model path
    print("Entering in the load model")
    model_path = './model/'
    print("Files in model directory:", os.listdir(model_path))
    model = tf.keras.models.load_model('./model/saved_model.h5',custom_objects=None , safe_mode = False)

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


