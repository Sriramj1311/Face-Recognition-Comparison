import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

_mtcnn = None
_model = None

def _models():
    global _mtcnn, _model
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device="cpu")
    if _model is None:
        _model = InceptionResnetV1(pretrained="vggface2").eval().to("cpu")
    return _mtcnn, _model

def _embed(img):
    mtcnn, model = _models()
    face = mtcnn(img)
    if face is None:
        raise ValueError("FaceNet could not detect a face.")
    with torch.no_grad():
        emb = model(face.unsqueeze(0)).cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return face.permute(1,2,0).numpy().clip(0,1), emb

def run(image, references):
    face_img, emb = _embed(image)
    matches=[]
    for name, ref in references:
        try:
            _, ref_emb = _embed(ref)
            sim=float(np.dot(emb, ref_emb))
            matches.append({"Reference":name, "Cosine similarity":round(sim,4)})
        except Exception:
            continue
    matches.sort(key=lambda x:x["Cosine similarity"], reverse=True)
    out=image.copy()
    message="Face embedding generated."
    if matches:
        best=matches[0]
        message=f"Best reference match: {best['Reference']} (cosine similarity {best['Cosine similarity']:.4f})"
    else:
        message="Embedding generated. Add reference faces to perform recognition."
    return {"ok":True,"image":out,"face":face_img,"embedding":emb,"matches":matches,
            "message":message,"summary":message}
