import streamlit as st
import requests

st.set_page_config(page_title="Deepfake Detection System", layout="wide")
st.title("Deepfake Detection System")

# Two options: Camera and Video
option = st.sidebar.selectbox("Choose Input Type", ["Camera", "Video"])

if option == "Camera":
    st.header("Upload a Photo")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        # Predict using the backend
        response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded_image.getvalue()})
        result = response.json()
        st.write("Prediction:", result['prediction'])
        st.write("Confidence:", result['confidence'])
        # Explainability: get Grad-CAM
        if st.button("Show Grad-CAM Explanation"):
            explanation = requests.post("http://127.0.0.1:5000/explain", files={"file": uploaded_image.getvalue()})
            st.image(explanation.content, caption="Grad-CAM Heatmap", use_column_width=True)
else:
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        # For video, you might want to process one frame or aggregate predictions.
        # For simplicity, let's extract one frame:
        response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded_video.getvalue()})
        result = response.json()
        st.write("Prediction:", result['prediction'])
        st.write("Confidence:", result['confidence'])
