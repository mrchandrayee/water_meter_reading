
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Optional
import logging,cv2,numpy as np
from src.common.model_registry import load_model_registry,require_model_path
from .detector import MeterFaceDetector,EfficientNetDigitOCR,SubDialReader,crop_meter_face,process_cropped_face
from .visualizer import WaterMeterVisualizer
logger=logging.getLogger(__name__)

@dataclass(frozen=True)
class WaterMeterPipelineConfig:
    ocr_model_path:Optional[str|Path]=None
    registry_path:Optional[str|Path]="models/registry.json"
    resize_factor:int=2
    blur_ksize:int=9
    clahe_clip:float=3.0
    threshold_value:int=130
    skip_frames:int=1
    device:str="cpu"

class WaterMeterPipeline:
    """Meter face detection -> full-resolution crop -> odometer OCR and sub-dial analysis."""
    def __init__(self,config:Optional[WaterMeterPipelineConfig]=None):
        self.config=config or WaterMeterPipelineConfig()
        registry=load_model_registry(self.config.registry_path)
        ocr_path=self.config.ocr_model_path
        if ocr_path is None:
            try: ocr_path=require_model_path(registry,"water_meter_ocr")
            except FileNotFoundError: ocr_path=None
        self.face_detector=MeterFaceDetector(self.config.blur_ksize,self.config.clahe_clip,
                                             self.config.threshold_value)
        self.ocr_model=EfficientNetDigitOCR(str(ocr_path) if ocr_path else None,self.config.device)
        self.sub_dial_reader=SubDialReader()
        self.frame_counter=0
        self.last_hud_data={"odometer text":"","dials":{}}

    def process_frame(self,frame:np.ndarray)->dict[str,Any]:
        empty={"face detected":False,"face cx":0,"face cy":0,"face r":0,
               "odometer text":"","dials":{},"vis crop":None,"crop coords":None}
        if frame is None or frame.size==0:return empty
        self.frame_counter+=1
        h,w=frame.shape[:2]; n=max(1,self.config.resize_factor)
        small=cv2.resize(frame,(max(1,w//n),max(1,h//n)))
        best=self.face_detector.detect(small).get("best")
        if best is None:return empty
        cx,cy,r=best["cx"]*n,best["cy"]*n,best["radius"]*n
        face,(x1,y1,x2,y2)=crop_meter_face(frame,cx,cy,r)
        if face.size==0:return empty
        vis,hud=process_cropped_face(face,self.ocr_model,self.sub_dial_reader)
        result={"face detected":True,"face cx":cx,"face cy":cy,"face r":r,
                "odometer text":hud.get("odometer text",""),"dials":hud.get("dials",{}),
                "vis crop":vis,"crop coords":(x1,y1,x2,y2)}
        self.last_hud_data=hud
        return result
