import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def get_gradcam_heatmap(model, image, last_conv_layer_name):
    # Create a sub-model for Grad-CAM
    grad_model = Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = predictions[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return output

if __name__ == "__main__":
    # 1) Load model
    model_path = r"C:\Users\gau68\DeepfakeDetector\models\deepfake_detector.h5"
    model = tf.keras.models.load_model(model_path, compile=False)

    # 2) Dummy forward pass to "build" the model
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    _ = model(dummy_input)  # Now model has defined outputs

    # 3) Now run Grad-CAM
    image_path = r"C:\Users\gau68\DeepfakeDetector\data_preprocessing\extracted_frames\real\01__exit_phone_room\frame_0.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        exit()

    # Resize to match model input
    image_resized = cv2.resize(image, (128, 128))
    last_conv_layer_name = "conv2d_1"

    # Generate heatmap
    heatmap = get_gradcam_heatmap(model, image_resized, last_conv_layer_name)

    # Overlay heatmap
    overlay = overlay_heatmap(heatmap, image_resized)
    cv2.imwrite("gradcam_output.jpg", overlay)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
