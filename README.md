# VisionLab — Comparative Face Detection & Recognition

A Streamlit computer-vision demo comparing **FaceNet, DeepFace, Template Matching, and Viola-Jones** on the same image.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a test image. Optionally add reference face images and a template image.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the complete contents of this folder.
3. Open Streamlit Community Cloud and choose **Deploy an app**.
4. Select your repository, branch, and `app.py`.
5. Deploy.

The first run can take time because pretrained models may need to download.

## Project structure

```text
app.py
content.py
requirements.txt
packages.txt
modules/
  facenet_module.py
  deepface_module.py
  template_matching.py
  viola_jones.py
database/reference_faces/
results/
```

> This project is intended as an educational demonstration. Facial attribute estimates and recognition results can be inaccurate.
