from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('tomato_model.h5')
labels_dir = "tomato_dataset/train"
class_labels = sorted(os.listdir(labels_dir))

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        try:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_idx]

            return jsonify({'predicted_class': predicted_class_label, 'accuracy': f"{np.max(predictions):.2f}"})
        except Exception as e:
            return jsonify({'error': e}), 500
        
    if request.method == 'GET':
        return jsonify({'message': 'Hello World!'})

if __name__ == '__main__':
    app.run(debug=True)
