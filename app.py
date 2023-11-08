import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# Remove spaces in the model file name
model = load_model("efficientnetb3-Eye Disease-94.55.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input1():
    return render_template('input.html')

@app.route('/predict', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)  # Use double underscores instead of spaces
        filepath = os.path.join(basepath, 'uploads', f.filename)  # Fix the folder name
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))  # Remove the '3' from target_size
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['Cataract','Diabetic _retinopathy','Glaucoma','Normal']
        #'Boletus', 'Lactarius', 'Russula'
         #'Normal','Uclerative-colitis','Polyps','Esophagitis'rasna
        #'Cataract','Diabetic _retinopathy','Glaucoma','Normal'
        result = str(index[prediction[0]])
        print(result)
        return render_template('output.html', prediction=result)

if __name__ == "__main__":  # Fix the if statement
    app.run(debug=True)





















