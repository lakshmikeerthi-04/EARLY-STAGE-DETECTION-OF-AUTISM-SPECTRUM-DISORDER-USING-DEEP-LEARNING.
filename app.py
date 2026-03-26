from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from mtcnn import MTCNN # type: ignore

detector=MTCNN()

app = Flask(__name__)
model = load_model(r"C:\Users\laksh\OneDrive\Desktop\ASD\model\asdmodel.keras")  # Your trained model

UPLOAD_FOLDER =r"C:\Users\laksh\OneDrive\Desktop\ASD\static\uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_human_face(image_path):
    #img = cv2.imread(image_path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    img=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    return len(faces) > 0

def predict_image(image_path):
    if not is_human_face(image_path):
        return "Given image is not human"

    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    return "Autistic" if prediction > 0.5 else "Non-Autistic"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        if "image" not in request.files:
            prediction = "No file part"
        else:
            file = request.files["image"]
            if file.filename == "":
                prediction = "No selected file"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                prediction = predict_image(filepath)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)