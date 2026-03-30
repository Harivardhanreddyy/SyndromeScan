from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector
from keras.models import load_model
from joblib import load
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import joblib
from sklearn.ensemble import StackingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
import sys
from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from werkzeug.utils import secure_filename

# Compatibility for Python 3.10+
if sys.version_info >= (3, 10):
    collections.Hashable = collections.abc.Hashable


app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='syndrome'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():

    return render_template('index.html')

@app.route('/about')
def about():

    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Confirm password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    
    return render_template('home.html')


# Load the pre-trained models
logistic_vgg16 = joblib.load('logistic_vgg16.joblib')
lgbm_vgg16 = joblib.load('lgbm_vgg16.joblib')
nmf_vgg16 = joblib.load('nmf_vgg16.joblib')
svm_mobilenet = joblib.load('svm_mobilenet.joblib')

# Define the VGG16 model for feature extraction
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg_feature_extractor = Model(inputs=vgg16_base.input, outputs=Flatten()(vgg16_base.output))

# Define the MobileNet model for feature extraction
mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_feature_extractor = Model(inputs=mobilenet_base.input, outputs=Flatten()(mobilenet_base.output))

# Path to save the uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process the image and make predictions
def predict_image(image_path, algorithm):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Ensure consistent image size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    if algorithm == 'vnl_net':
        # Extract features using VGG16
        vgg_features = vgg_feature_extractor.predict(img_array)
        nmf_features = nmf_vgg16.transform(vgg_features)
        enhanced_features = lgbm_vgg16.predict_proba(nmf_features)[:, 1].reshape(-1, 1)
        pred = logistic_vgg16.predict(enhanced_features)
    elif algorithm == 'mobilenet_svm':
        # Extract features using MobileNet
        mobilenet_features = mobilenet_feature_extractor.predict(img_array)
        pred = svm_mobilenet.predict(mobilenet_features)
    else:
        return "Invalid Algorithm Selection"

    # Return the class label
    if pred == 1:
        return "Normal Kid"
    else:
        return "Has a Down Syndrome"

@app.route('/algorithm', methods=["GET", "POST"])
def algorithm():
    if request.method == "POST":
        if 'file' not in request.files or 'algorithm' not in request.form:
            return redirect(request.url)
        file = request.files['file']
        algorithm = request.form['algorithm']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file securely
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the class using the selected algorithm
            prediction = predict_image(file_path, algorithm)

            # Pass the prediction and image to the frontend
            return render_template('algorithm.html', prediction=prediction, image_name=filename)

    return render_template('algorithm.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


@app.route('/graph')
def graph():
    return render_template('graph.html')


if __name__ == '__main__':
    app.run(debug = True)
    