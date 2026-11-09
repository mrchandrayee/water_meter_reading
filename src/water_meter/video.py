
from __future__ import annotations
import os,time,cv2,numpy as np
from pathlib import Path
from .visualizer import WaterMeterVisualizer

def show_frame(window_name,frame,wait_ms=30,display_size=(960,540)):
    shown=cv2.resize(frame,display_size) if display_size else frame
    cv2.imshow(window_name,shown); key=cv2.waitKey(wait_ms)&0xFF
    if key in (ord("p"),): cv2.waitKey(0)
    return key not in (27,ord("q"))

def draw_on_frame(frame,pipeline_result,hud_data=None):
    display=frame.copy()
    crop=pipeline_result.get("vis crop"); coords=pipeline_result.get("crop coords")
    if crop is not None and coords is not None:
        x1,y1,x2,y2=coords
        if x2>x1 and y2>y1: display[y1:y2,x1:x2]=cv2.resize(crop,(x2-x1,y2-y1))
    if pipeline_result.get("face detected"):
        WaterMeterVisualizer.draw_tech_hud(display,pipeline_result["face cx"],
            pipeline_result["face cy"],pipeline_result["face r"],hud_data or {})
    return display

def write_video(pipeline,source_path,target_path,max_frames=None,progress=True,
                progress_interval=1,imshow=False,window_name="Water Meter Reading",
                wait_ms=30,display_size=(960,540)):
    cap=cv2.VideoCapture(source_path)
    if not cap.isOpened():raise RuntimeError(f"Could not open video source: {source_path}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:total=min(total,max_frames) if total else max_frames
    Path(target_path).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(target_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():cap.release();raise RuntimeError(f"Cannot create output: {target_path}")
    idx=0; start=time.time(); fps_count=0; fps_timer=start; current_fps=0.; last_reading=""
    skip=max(1,pipeline.config.skip_frames)
    try:
        while True:
            ok,frame=cap.read()
            if not ok or (max_frames is not None and idx>=max_frames):break
            idx+=1
            if (idx-1)%skip:
                writer.write(frame);continue
            t0=time.time(); result=pipeline.process_frame(frame)
            reading=result.get("odometer text","")
            if reading:last_reading=reading
            vis=draw_on_frame(frame,result,pipeline.last_hud_data)
            fps_count+=1
            if time.time()-fps_timer>=1:
                current_fps=fps_count/(time.time()-fps_timer);fps_count=0;fps_timer=time.time()
            WaterMeterVisualizer.draw_info(vis,current_fps,last_reading)
            writer.write(vis)
            if progress and (progress_interval<=1 or idx%progress_interval==0):
                print(f"{idx}/{total or '?'} | {1000*(time.time()-t0):.1f} ms | "
                      f"face={'FOUND' if result.get('face detected') else 'NOT FOUND'} | "
                      f"odometer={reading or 'N/A'}")
            if imshow and not show_frame(window_name,vis,wait_ms,display_size):
                print("Display stopped by user.");break
    finally:
        cap.release();writer.release()
        if imshow:cv2.destroyAllWindows()
    if progress:
        print(f"Saved annotated video: {target_path}")
        if last_reading:print(f"Last odometer reading: {last_reading}")
