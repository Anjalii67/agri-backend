#  Agri – AI Crop Disease Detection using MobilePlantViT

<div align="center">

#  Agri

### AI-Powered Crop Disease Detection using MobilePlantViT

AI-powered crop disease detection application using MobilePlantViT with a Flutter frontend and Flask backend.

<br>

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter)](https://flutter.dev)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![Android](https://img.shields.io/badge/Platform-Android-success?logo=android)]()

<br>

[![Watch Demo](https://img.shields.io/badge/▶️-Watch%20Demo-success?style=for-the-badge)](https://drive.google.com/file/d/1FUja-t0-dpRJjRisFOhxhruw1Ph3WCCv/view?usp=sharing)

</div>

---

#  Problem Statement

Crop diseases significantly reduce agricultural productivity, and timely diagnosis is often requires expert knowledge that may not always be available to farmers.

Agri provides an AI-powered solution that enables farmers to detect crop diseases directly from leaf images using a custom **MobilePlantViT** deep learning model. The application delivers fast predictions through an intuitive Flutter interface backed by a Flask REST API.

---

#  Features

-  Capture images using Camera or Gallery
-  Supports **Tomato** and **Corn** crops
-  AI-powered disease detection using MobilePlantViT
-  🇮🇳 Bilingual interface (English & Hindi)
-  Confidence score for every prediction
-  Low-confidence warning for uncertain predictions
-  Flutter Android application
-  Flask REST API backend

---

#  Demo

Click below to watch the application in action.

### ▶️ https://drive.google.com/file/d/1FUja-t0-dpRJjRisFOhxhruw1Ph3WCCv/view?usp=sharing

> **Note**
>
> The current version runs locally. Cloud deployment and support for additional crops are planned in future releases.

---

#  Architecture

```text
Flutter Android App
        │
        ▼
Camera / Gallery
        │
        ▼
Flask REST API
        │
        ▼
Image Preprocessing
        │
        ▼
MobilePlantViT Model
        │
        ▼
Disease Prediction
        │
        ▼
JSON Response
        │
        ▼
Flutter Result Screen
```

---

#  Why MobilePlantViT?

MobilePlantViT was selected because it combines the efficiency of lightweight Convolutional Neural Networks (CNNs) with the contextual understanding of Vision Transformers.

This architecture provides:

- High accuracy while remaining lightweight
- Faster inference suitable for mobile applications
- Better feature extraction from complex leaf disease patterns
- Efficient deployment for real-world agricultural use cases

---

#  Model Pipeline

1. Capture or upload a crop leaf image.
2. Preprocess and normalize the image.
3. Perform inference using the MobilePlantViT model.
4. Compute prediction confidence.
5. Return the predicted disease along with the confidence score through the REST API.

---

#  Dataset

Supported Crops

-  Tomato (10 Classes)
-  Corn (4 Classes)

**Total Disease Classes:** **14**

---

#  Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **97.08%** |
| Supported Crops | Tomato & Corn |
| Disease Classes | 14 |
| Deep Learning Model | MobilePlantViT |

---

#  Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | PyTorch |
| Backend | Flask |
| Frontend | Flutter |
| Programming Languages | Python, Dart |
| Image Processing | Pillow, Torchvision |

---

#  Getting Started

## Clone Repository

```bash
git clone https://github.com/Anjalii67/agri-backend.git
cd agri-backend
```

## Backend Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

## Flutter Setup

```bash
cd agri_app

flutter pub get

flutter run
```

Ensure both the backend server and mobile device are connected to the same Wi-Fi network.

---

#  Project Structure

```text
agri-backend/
│
├── app.py
├── requirements.txt
├── class_names.txt
├── model/
│
└── agri_app/
    ├── android/
    ├── ios/
    ├── lib/
    ├── pubspec.yaml
```

---

#  API Reference

## POST /predict

Upload a crop leaf image in **JPG** or **PNG** format.

### Sample Response

```json
{
  "crop": "Tomato",
  "disease": "Early Blight",
  "confidence": 96.5
}
```

---

#  Future Improvements

- Support additional crop varieties
- Offline inference using TensorFlow Lite
- Disease treatment recommendations
- Farmer dashboard
- GPS-based disease monitoring
- Weather-based disease prediction
- Voice assistance in regional languages

---

