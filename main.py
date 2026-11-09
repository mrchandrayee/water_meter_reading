
from __future__ import annotations
import argparse, logging
from pathlib import Path
import cv2
from src.water_meter.pipeline import WaterMeterPipeline, WaterMeterPipelineConfig
from src.water_meter.video import write_video, draw_on_frame, show_frame

def main():
    p=argparse.ArgumentParser(description="Water Meter Reading")
    p.add_argument("--video", help="Input video path")
    p.add_argument("--image", help="Input image path")
    p.add_argument("--source", default="0", help="Webcam index or video path")
    p.add_argument("--output", default="outputs/output_water_meter.mp4")
    p.add_argument("--mapping", default="models/registry.json")
    p.add_argument("--ocr-model", default=None)
    p.add_argument("--resize-factor", type=int, default=2)
    p.add_argument("--blur-ksize", type=int, default=9)
    p.add_argument("--clahe-clip", type=float, default=3.0)
    p.add_argument("--threshold", type=int, default=130)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--imshow", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg=WaterMeterPipelineConfig(ocr_model_path=a.ocr_model,registry_path=a.mapping,
        resize_factor=max(1,a.resize_factor),blur_ksize=a.blur_ksize,
        clahe_clip=a.clahe_clip,threshold_value=a.threshold,skip_frames=max(1,a.skip),
        device=a.device)
    pipe=WaterMeterPipeline(cfg)
    if a.image:
        frame=cv2.imread(a.image)
        if frame is None: raise SystemExit(f"Cannot read image: {a.image}")
        result=pipe.process_frame(frame)
        vis=draw_on_frame(frame,result,pipe.last_hud_data)
        out=Path(a.output).with_suffix(".jpg"); out.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(out),vis); print(f"Saved: {out}")
        if a.imshow: show_frame("Water Meter Reading",vis)
    else:
        src=a.video if a.video else (int(a.source) if a.source.isdigit() else a.source)
        write_video(pipe,src,a.output,max_frames=a.max_frames,
                    progress=not a.no_progress,imshow=a.imshow)
if __name__=="__main__": main()
