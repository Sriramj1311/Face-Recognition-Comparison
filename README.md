# VisionLab — Comparative Face Detection & Recognition

A Streamlit-based computer vision application that demonstrates and compares four different face detection and recognition approaches on the same input image:

- **FaceNet**
- **DeepFace**
- **Template Matching**
- **Viola-Jones**

The project is designed as an educational demonstration to understand how different computer vision techniques perform for face detection and recognition.

---

## 🚀 Live Demo

**Streamlit App:**  
git clone https://github.com/Sriramj1311/Face-Recognition-Comparison.git

> Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit Community Cloud URL.

---

## 📌 Features

- Upload a test image through an interactive Streamlit interface.
- Detect and recognize faces using **FaceNet**.
- Perform face analysis and recognition using **DeepFace**.
- Compare faces using **Template Matching**.
- Detect faces using the classical **Viola-Jones** algorithm.
- View the results of all methods in a single application.
- Compare the methods based on their characteristics and applications.
- Includes educational information and viva questions related to the implemented methods.

---

## 🧠 Methods Used

### 1. FaceNet

FaceNet is a deep-learning-based face recognition method that represents faces as numerical embeddings.

It maps a face image into a feature vector, allowing faces to be compared using the similarity between their embeddings.

**Advantages:**
- High recognition accuracy
- Effective for face verification and identification
- Uses deep-learning-based feature representations

---

### 2. DeepFace

DeepFace is a deep-learning framework for face analysis and recognition.

It provides functionality for face verification and recognition using pretrained deep-learning models.

**Advantages:**
- Supports multiple face recognition models
- Provides powerful face analysis capabilities
- Suitable for modern face recognition applications

---

### 3. Template Matching

Template Matching is a traditional computer vision technique that searches for a smaller template image within a larger image.

In this project, it is used to demonstrate a simple image-based matching approach.

**Advantages:**
- Simple to understand
- Easy to implement
- Does not require deep-learning models

**Limitations:**
- Sensitive to scale, rotation, and lighting changes
- Less suitable for complex face recognition tasks

---

### 4. Viola-Jones

Viola-Jones is a classical object detection algorithm commonly used for face detection.

It uses Haar-like features, an integral image, and a cascade classifier to efficiently detect faces.

**Advantages:**
- Fast detection
- Lightweight
- Works well for basic frontal-face detection
- Does not require deep-learning models

---

## 📊 Comparison

| Method | Type | Main Purpose | Deep Learning | Speed |
|---|---|---|---|---|
| FaceNet | Deep Learning | Face Recognition | Yes | Moderate |
| DeepFace | Deep Learning | Face Analysis & Recognition | Yes | Moderate |
| Template Matching | Classical CV | Image/Template Matching | No | Fast |
| Viola-Jones | Classical CV | Face Detection | No | Very Fast |

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Pandas**
- **TensorFlow**
- **Keras**
- **PyTorch**
- **FaceNet-PyTorch**
- **DeepFace**

---

## 📁 Project Structure

```text
Face-Recognition-Comparison/
│
├── app.py
├── content.py
├── requirements.txt
├── packages.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── facenet_module.py
│   ├── deepface_module.py
│   ├── template_matching.py
│   └── viola_jones.py
│
├── database/
│   └── reference_faces/
│       └── .gitkeep
│
└── results/
    └── .gitkeep
