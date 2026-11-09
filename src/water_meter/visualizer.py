
from __future__ import annotations
import cv2
class WaterMeterVisualizer:
    @staticmethod
    def draw_tech_hud(frame,cx,cy,radius,hud_data):
        cv2.circle(frame,(int(cx),int(cy)),int(radius),(0,255,0),2,cv2.LINE_AA)
        cv2.circle(frame,(int(cx),int(cy)),4,(0,255,0),-1)
        x,y=12,30
        cv2.rectangle(frame,(x,y),(min(frame.shape[1]-8,frame.shape[1]-x),y+78),(10,15,10),-1)
        cv2.putText(frame,"WATER METER READING",(x+10,y+20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
        odo=hud_data.get("odometer text") or "N/A"
        cv2.putText(frame,f"MAIN ODOMETER: {odo}",(x+10,y+48),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,255,0),2)
        dials=hud_data.get("dials",{})
        labels="  ".join(f"{k}: {v.get('value') if v.get('value') is not None else '--'}" for k,v in dials.items())
        cv2.putText(frame,labels[:90],(x+10,y+68),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,220,0),1)
        return frame
    @staticmethod
    def draw_info(frame,fps,reading):
        text=f"FPS: {fps:.1f} | Reading: {reading or 'N/A'}"
        cv2.putText(frame,text,(12,frame.shape[0]-14),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)
        return frame
