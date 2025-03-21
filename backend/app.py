from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model(r"C:\Users\gau68\DeepfakeDetector\models\deepfake_detector.h5")

def preprocess_image(file):
    # Convert uploaded file to image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"})
    file = request.files['file']
    img = preprocess_image(file)
    prediction = model.predict(img)[0][0]
    result = "Real" if prediction < 0.5 else "Deepfake"
    return jsonify({"prediction": result, "confidence": float(prediction)})

@app.route('/explain', methods=['POST'])
def explain():
    # For explainability, return Grad-CAM heatmap image
    from utils.explainability import get_gradcam_heatmap, overlay_heatmap
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"})
    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (128, 128))
    
    heatmap = get_gradcam_heatmap(model, img_resized, last_conv_layer_name='conv2d_1')
    output_img = overlay_heatmap(heatmap, img_resized)
    output_path = "backend/gradcam_output.jpg"
    cv2.imwrite(output_path, output_img)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
