import numpy as np
import cv2
from deepface import DeepFace

MODEL="Facenet512"

def _tmp(img):
    bgr=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr

def run(image, references):
    # Analyze the uploaded image first. DeepFace downloads its selected model on first use.
    analysis = DeepFace.analyze(_tmp(image), actions=["age","gender","emotion"], detector_backend="opencv", enforce_detection=True)
    if isinstance(analysis,list):
        analysis=analysis[0]
    region=analysis.get("region",{})
    x,y,w,h=[int(region.get(k,0)) for k in ("x","y","w","h")]
    crop=image[max(0,y):max(0,y+h), max(0,x):max(0,x+w)]
    if crop.size==0:
        crop=image
    matches=[]
    for name, ref in references:
        try:
            result=DeepFace.verify(_tmp(image), _tmp(ref), model_name=MODEL,
                                   detector_backend="opencv", enforce_detection=True)
            matches.append({"Reference":name,"Verified":bool(result["verified"]),
                            "Distance":round(float(result["distance"]),4),
                            "Threshold":round(float(result["threshold"]),4)})
        except Exception:
            continue
    matches.sort(key=lambda z:z["Distance"])
    if matches:
        best=matches[0]
        message=f"Closest reference: {best['Reference']} — {'verified' if best['Verified'] else 'not verified'}."
    else:
        message="DeepFace analysis completed. Add reference faces for identity verification."
    return {"ok":True,"image":image,"face":crop,"model":MODEL,
            "matches":matches,"attributes":{k:analysis.get(k) for k in ("age","gender","emotion")},
            "message":message,"summary":message}
