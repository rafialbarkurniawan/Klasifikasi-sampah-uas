from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from model_loader import model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Menentukan direktori tempat gambar akan disimpan
UPLOAD_FOLDER = os.path.abspath('./uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max upload size to 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {file_path}")
        try:
            file.save(file_path)
            print("File saved successfully")
            prediction, predicted_class_name = predict_image(file_path)
            return render_template('result.html', prediction=prediction, image_url=url_for('uploaded_file', filename=filename), predicted_class_name=predicted_class_name)
        except Exception as e:
            print(f"Error saving file: {e}")
            return redirect(request.url)
    else:
        print("File type not allowed")
        return redirect(request.url)


def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    class_names = {'BotolKaca': 0, 'BotolPlastik': 1, 'GelasDisposable': 2, 'Kaleng': 3, 'Kardus': 4, 'WadahKaca': 5}
    predicted_class_name = [name for name, index in class_names.items() if index == predicted_class_index][0]
    return prediction, predicted_class_name

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
