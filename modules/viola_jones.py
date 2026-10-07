import cv2
import os

def run(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    cascade_path=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(cascade_path)
    faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
    out=image.copy()
    boxes=[]
    for x,y,w,h in faces:
        cv2.rectangle(out,(x,y),(x+w,y+h),(50,220,80),3)
        boxes.append((int(x),int(y),int(w),int(h)))
    msg=f"Viola-Jones detected {len(boxes)} face(s)."
    return {"ok":True,"image":out,"count":len(boxes),"boxes":boxes,
            "message":msg,"summary":msg}
