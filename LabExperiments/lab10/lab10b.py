#Generate Artificial Video
import os
import numpy as np
import cv2 as cv

def generatebox_mp4(filename="test_box.mp4",num_frames=120, framerate=30):
    if os.path.exists(filename):
        os.remove(filename)

    width,height=640,480
    fourcc=cv.VideoWriter_fourcc(*'mp4v')
    out=cv.VideoWriter(filename,fourcc,framerate,(width,height))
    if not out.isOpened():
        print("Created file not opened ")
        return None

    #base_frame is black background
    base_frame=np.zeros((height,width,3),dtype=np.uint8)
    box_size=80
    for i in range(num_frames):
        frame=base_frame.copy()
        x=50+int(i*1.0)
        y=100+int(i*0.5)
        cv.rectangle(frame,(x,y),(x+box_size,y+box_size),(255,255,255),-1)
        out.write(frame)
    out.release()

    if os.path.exists(filename) and os.path.getsize(filename)>0:
        print("File created successfully")

generatebox_mp4()

    
