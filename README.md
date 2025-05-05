"# Eggplant-Pest-and-Disease-Prediction" 

This Streamlit web app allows users to upload images of eggplant leaves and get real-time predictions of pest or disease types. It uses a YOLOv11s object detection model trained to identify common issues in eggplants, and provides the cause and remedy for each detected class.

---

## 🌐 Live App

👉 [Click here to use the dashboard]([https://your-streamlit-link.streamlit.app/](https://eggplant-pest-and-disease-prediction-1.streamlit.app/))  

---

## 🖼️ Features

- Upload images of eggplant leaves
- Get bounding boxes around detected pests/diseases
- Display prediction labels with confidence scores
- View cause and remedy from a metadata file

---

## 📁 Repository Contents

| File                          | Description                                      |
|------------------------------|--------------------------------------------------|
| `app.py`                     | Main Streamlit application code                  |
| `best_eggplant_model_3.pt`   | YOLOv11s trained model (pests and diseases)        |
| `Eggplant_Cause_and_Remedy.json` | Metadata with cause and remedy per class   |
| `requirements.txt`           | Python dependencies for the app                  |
