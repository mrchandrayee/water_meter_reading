
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

class MeterFaceDetector:
    """Detect a circular meter face using CLAHE, blur and Hough circles."""
    def __init__(self, blur_ksize:int=9, clahe_clip:float=3.0, threshold_value:int=130):
        self.blur_ksize=max(3,int(blur_ksize)|1)
        self.clahe_clip=float(clahe_clip)
        self.threshold_value=int(threshold_value)

    def detect(self, frame:np.ndarray)->dict:
        if frame is None or frame.size==0: return {"best":None,"candidates":[]}
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
        clahe=cv2.createCLAHE(clipLimit=self.clahe_clip,tileGridSize=(8,8))
        enhanced=clahe.apply(gray)
        blurred=cv2.GaussianBlur(enhanced,(self.blur_ksize,self.blur_ksize),1.5)
        h,w=gray.shape[:2]; min_dim=min(h,w)
        circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.2,
            minDist=max(30,min_dim//3),param1=100,param2=35,
            minRadius=max(20,min_dim//8),maxRadius=max(25,min_dim//2))
        candidates=[]
        if circles is not None:
            for x,y,r in np.round(circles[0]).astype(int):
                if x-r<0 or y-r<0 or x+r>w or y+r>h: continue
                # Prefer large, central circles.
                center_penalty=((x-w/2)**2+(y-h/2)**2)**.5/max(1,min_dim)
                score=float(r/min_dim)-0.15*center_penalty
                candidates.append({"cx":int(x),"cy":int(y),"radius":int(r),"score":score})
        candidates.sort(key=lambda z:z["score"],reverse=True)
        return {"best":candidates[0] if candidates else None,"candidates":candidates}

def crop_meter_face(frame:np.ndarray,cx:int,cy:int,radius:int,margin:float=.08):
    h,w=frame.shape[:2]; r=int(radius*(1+margin))
    x1=max(0,cx-r); y1=max(0,cy-r); x2=min(w,cx+r); y2=min(h,cy+r)
    return frame[y1:y2,x1:x2].copy(),(x1,y1,x2,y2)

class EfficientNetDigitOCR:
    """Optional EfficientNet-B0 digit classifier; falls back to Tesseract OCR."""
    def __init__(self,model_path:Optional[str]=None,device:str="cpu"):
        self.model=None; self.device=device; self.classes=[str(i) for i in range(10)]
        if model_path and Path(model_path).exists():
            try:
                import torch
                import torch.nn as nn
                from torchvision.models import efficientnet_b0
                checkpoint=torch.load(model_path,map_location=device)
                classes=checkpoint.get("classes",self.classes) if isinstance(checkpoint,dict) else self.classes
                state=checkpoint.get("state_dict",checkpoint) if isinstance(checkpoint,dict) else checkpoint
                model=efficientnet_b0(weights=None)
                model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(classes))
                model.load_state_dict(state); model.to(device).eval()
                self.model=model; self.classes=list(classes)
            except Exception as e:
                raise RuntimeError(f"Could not load EfficientNet OCR weights: {e}") from e

    def predict_digits(self, image:np.ndarray)->str:
        if image is None or image.size==0: return ""
        if self.model is not None:
            try:
                import torch
                from PIL import Image
                from torchvision import transforms
                # Digit-strip classifier should be trained to classify a complete 5-digit strip.
                tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                tensor=tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                with torch.no_grad(): idx=int(self.model(tensor).argmax(1).item())
                return str(self.classes[idx])
            except Exception: pass
        try:
            import pytesseract
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
            gray=cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
            gray=cv2.createCLAHE(2.0,(8,8)).apply(gray)
            text=pytesseract.image_to_string(gray,config="--psm 7 -c tessedit_char_whitelist=0123456789")
            return "".join(ch for ch in text if ch.isdigit())
        except Exception:
            return ""

class SubDialReader:
    """Estimate analog sub-dial pointer position using thresholding and PCA."""
    def read(self,crop:np.ndarray)->dict:
        if crop is None or crop.size==0: return {"value":None,"angle":None}
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape[:2]
        blur=cv2.GaussianBlur(gray,(5,5),0)
        _,mask=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # Restrict analysis to pixels near the dial centre, excluding the outer rim.
        yy,xx=np.where(mask>0)
        if len(xx)<10: return {"value":None,"angle":None}
        cx,cy=w/2,h/2
        dist=np.sqrt((xx-cx)**2+(yy-cy)**2)
        keep=(dist>min(h,w)*.08)&(dist<min(h,w)*.43)
        pts=np.column_stack((xx[keep],yy[keep])).astype(np.float32)
        if len(pts)<8: return {"value":None,"angle":None}
        pts-=pts.mean(axis=0)
        cov=np.cov(pts.T); vals,vecs=np.linalg.eigh(cov)
        axis=vecs[:,int(np.argmax(vals))]
        angle=float(np.degrees(np.arctan2(-axis[1],axis[0]))%180)
        return {"value":round(angle/180*10,2),"angle":angle}

def process_cropped_face(face_crop:np.ndarray,ocr_model:EfficientNetDigitOCR,
                         sub_dial_reader:SubDialReader):
    """Read odometer strip and estimate the three small pointer dials."""
    h,w=face_crop.shape[:2]
    # Odometer window is typically in the upper central area of the meter face.
    odometer_roi=face_crop[int(h*.16):int(h*.43),int(w*.20):int(w*.82)]
    odometer_text=ocr_model.predict_digits(odometer_roi)
    # Three sub-dials are arranged across the lower half on common meter layouts.
    dial_rois={}
    for name,(cx,cy) in {"x0.01":(.28,.69),"x0.001":(.57,.69),"x0.0001":(.78,.69)}.items():
        rw=int(w*.22); rh=int(h*.28); x=int(w*cx-rw/2); y=int(h*cy-rh/2)
        dial_rois[name]=face_crop[max(0,y):min(h,y+rh),max(0,x):min(w,x+rw)]
    dials={name:sub_dial_reader.read(roi) for name,roi in dial_rois.items()}
    vis=face_crop.copy()
    cv2.rectangle(vis,(int(w*.20),int(h*.16)),(int(w*.82),int(h*.43)),(0,180,255),2)
    for name,(cx,cy) in {"x0.01":(.28,.69),"x0.001":(.57,.69),"x0.0001":(.78,.69)}.items():
        cv2.ellipse(vis,(int(w*cx),int(h*cy)),(int(w*.11),int(h*.14)),0,0,360,(255,0,255),2)
        val=dials[name]["value"]
        cv2.putText(vis,f"{name}: {val if val is not None else '--'}",(max(2,int(w*cx-w*.1)),min(h-5,int(h*cy+h*.17))),
                    cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,0),1,cv2.LINE_AA)
    cv2.putText(vis,f"ODOMETER: {odometer_text or 'N/A'}",(8,22),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2,cv2.LINE_AA)
    return vis,{"odometer text":odometer_text,"dials":dials}
