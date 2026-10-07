import cv2
import numpy as np

def run(image, template):
    if template is None:
        return {"ok":False,"error":"Upload a template image in the sidebar to run Template Matching."}
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    tgray=cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)
    if tgray.shape[0]>gray.shape[0] or tgray.shape[1]>gray.shape[1]:
        return {"ok":False,"error":"Template must be smaller than the uploaded image."}
    result=cv2.matchTemplate(gray,tgray,cv2.TM_CCOEFF_NORMED)
    _,score,_,loc=cv2.minMaxLoc(result)
    h,w=tgray.shape
    out=image.copy()
    cv2.rectangle(out,loc,(loc[0]+w,loc[1]+h),(255,80,40),3)
    found=score>=0.70
    msg=f"Best template match score: {score:.3f}."
    return {"ok":True,"image":out,"template":template,"score":float(score),
            "location":loc,"found":found,"message":msg,"summary":msg}
