import streamlit as st 
from ultralytics import YOLO 
import cv2 
import numpy as np 
import json
import os

# Load YOLO model
model = YOLO("best_eggplant_model_3.pt")  # Make sure best_eggplant_model_3.pt is in the same folder as app.py

# Load cause and remedy metadata
with open("Eggplant_Cause_and_Remedy.json", "r") as f:
    metadata = json.load(f)

# Streamlit UI
st.set_page_config(page_title="Eggplant Pest and Disease Detection Dashboard", page_icon="🍆", layout="centered")
st.title("🍆 Eggplant Pest and Disease Prediction")
st.write("Upload an eggplant leaf image to detect pests or diseases, and view the cause and remedy!")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image_bytes = uploaded_file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Make a copy for drawing
    img_result = img.copy()

    # Display the uploaded image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Inference
    st.subheader("🔍 Prediction Result")
    results = model.predict(img, conf=0.5)

    # Get boxes and labels
    boxes = results[0].boxes
    labels = results[0].names

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls)
            label_name = labels[cls_id]
            conf = box.conf.item()

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label text above the box
            label_text = f"{label_name} ({conf:.2f})"
            cv2.putText(img_result, label_text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display label name separately below
            st.success(f"**Prediction:** {label_name} (Confidence: {conf:.2f})")

            # Show cause and remedy if available
            if label_name in metadata:
                st.write(f"**Cause:** {metadata[label_name]['cause']}")
                st.write(f"**Remedy:** {metadata[label_name]['remedy']}")
            else:
                st.warning("⚠️ No metadata available for this class.")
        
        # Display the result image with all boxes drawn
        st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

    else:
        st.error("No pests or diseases detected. Try another image.")
