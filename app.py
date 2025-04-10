from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('jaundice_prediction_model.keras')

# Function to prepare the image
def prepare_image(uploaded_image):
    try:
        img = np.array(uploaded_image.convert('RGB'))  # Convert to RGB
        img = cv2.resize(img, (64, 64))  # Resize to model's expected input size
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        return None

# Function to process image and return base64 representation
def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Prediction function
def predict_image(image_array):
    prediction = model.predict(image_array)
    return 'NORMAL' if prediction[0] > 0.5 else 'JAUNDICED'

# Route for home page
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        try:
            img = Image.open(io.BytesIO(file.read()))
            img_array = prepare_image(img)

            if img_array is None:
                return jsonify({"error": "Invalid image format"}), 400

            result = predict_image(img_array)
            
            # Convert original image to OpenCV format and encode
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            encoded_img = encode_image(img_cv)

            return jsonify({"prediction": result, "processed_image": encoded_img})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
