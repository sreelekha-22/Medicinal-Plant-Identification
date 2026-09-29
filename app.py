# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# Trained by ML_projects/Med_Plant_Detection/mePD2.py on the FMLd dataset
# (80 Ayurvedic medicinal plant species). Resolved relative to this file so the
# app runs from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_2_vgg19.h5')

# Output order of Dense(80, softmax) -- must stay in step with the training script
CLASS_NAMES = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'ashoka', 'Astma_weed',
    'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami',
    'Bringaraja', 'camphor', 'Caricature', 'Castor', 'Catharanthus', 'Chakte',
    'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)',
    'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus',
    'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine',
    'kamakasturi', 'Kambajala', 'Kasambruga', 'kepala', 'Kohlrabi', 'Lantana',
    'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango',
    'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion',
    'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper',
    'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota',
    'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma',
    'Thumbe', 'Tomato', 'Tulsi', 'Turmeric',
]

# Load your trained model
model = load_model(MODEL_PATH)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError(
        'Model expects %d classes but CLASS_NAMES has %d entries.'
        % (model.output_shape[-1], len(CLASS_NAMES))
    )




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)
    preds = model.predict(x)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100.0

    return {
        'class_index': idx,
        'class_name': CLASS_NAMES[idx],
        'confidence': round(confidence, 2),
    }


    # preds = model.predict(x)
    # preds=np.argmax(preds, axis=1)
    # if preds==0:
    #     preds="The Person is Infected With Pneumonia"
    # else:
    #     preds="The Person is not Infected With Pneumonia"
    
    
    # return preds
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files.get('file')
        if f is None or f.filename == '':
            return {'error': 'No file uploaded. Attach an image under the "file" field.'}, 400

        # Save the file to ./uploads
        basepath = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        return model_predict(file_path, model)
    return None


if __name__ == '__main__':
    app.run(debug=True)
