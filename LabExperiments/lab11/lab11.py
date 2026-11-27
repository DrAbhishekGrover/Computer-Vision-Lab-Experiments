import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import cv2

#device=torch.device("gpu") if torch.cuda.is_available else torch.device("cpu")
device=torch.device("mps") if torch.cuda.is_available else torch.device("cpu")

#Load Pre-trained model
model_name="facebook/detr-resnet-50"
processor=DetrImageProcessor.from_pretrained(model_name)
model=DetrForObjectDetection.from_pretrained(model_name)
model.eval()
model.to(device)

dummy=np.random.randint(0,255,(800,800,3),dtype=np.uint8)
dummy_img=Image.fromarray(dummy)

inputs=processor(images=dummy_img,return_tensors="pt").to(device)
with torch.no_grad():
    outputs=model(**inputs)

#Get coordinates in [xmin,ymin,xmax,ymax] format
target_sizes=torch.tensor([dummy_img.size[::-1]],device=device)#Heignt,width
results=processor.post_process_object_detection(outputs,target_sizes=target_sizes,
                                                threshold=0.7)[0]
print(f"Detected boxes:{results['boxes']}")


#Develop a frame reading pipeline where
#Object will be detected in each frame
all_frames=[]
def run_detectionloop(video_path, model,processor,device,conf_threshold=0.7,
                      max_frames=150):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("file not opened")
        return None
    frame_idx=0
    #Creating Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter("outputlab11.avi", fourcc, fps, (width, height)) 
    #Video Writer Created   
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            print("End of Video reached")
            break

        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_pil=Image.fromarray(rgb_frame)
        height,width=frame.shape[:2]

        inputs=processor(images=frame_pil,return_tensors="pt").to(device)

        #Model Inference
        with torch.no_grad():
            outputs=model(**inputs)

        target_sizes=torch.tensor([[height,width]],device=device)
        results=processor.post_process_object_detection(
            outputs, target_sizes=target_sizes,
            threshold=conf_threshold
        )[0]

        filtered_boxes=results["boxes"].cpu().numpy()
        filtered_scores=results["scores"].cpu().numpy()
        filtered_labels=results["labels"].cpu().numpy()

        #Drawing boxes
        vis_frame = frame.copy()


        #id2label = getattr(model.config, "id2label", {})
        for box, score, lbl_id in zip(filtered_boxes, filtered_scores, filtered_labels):
            xmin, ymin, xmax, ymax = box
            pt1 = (int(np.clip(xmin, 0, width - 1)), int(np.clip(ymin, 0, height - 1)))
            pt2 = (int(np.clip(xmax, 0, width - 1)), int(np.clip(ymax, 0, height - 1)))

            lbl_int = int(lbl_id)
            rng = np.random.default_rng(lbl_int * 31 + 11)
            color = tuple(int(c) for c in rng.integers(50, 255, size=3))

            cv2.rectangle(vis_frame, pt1, pt2, color, 2)

            

        if writer is not None:
            writer.write(vis_frame)

        cv2.imshow("Detection",vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stopped by user ('q' pressed).")
            break

        all_frames.append({
            "frame_idx":frame_idx,
            "boxes":filtered_boxes,
            "scores":filtered_scores,
            "labels":filtered_labels
        })
        frame_idx=frame_idx+1
        if frame_idx>max_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    return all_frames

run_detectionloop("match5-c1.avi",model,processor,device)

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return float(inter / (a1 + a2 - inter))


def track_and_save(video_path, output_path, filtered_boxes, iou_thresh=0.3):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    tracks, next_id = {}, 1  # {id: box}
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        vis_frame = frame.copy()
        if not ret or idx >= len(filtered_boxes):
            break

        boxes = np.asarray(filtered_boxes[idx], dtype=np.float32).reshape(-1, 4)
        new_tracks, matched_boxes = {}, set()

        for tid, tbox in tracks.items():
            best_i, best_iou = -1, iou_thresh
            for i, b in enumerate(boxes):
                if i not in matched_boxes:
                    val = iou(tbox, b)
                    if val >= best_iou:
                        best_iou, best_i = val, i
            if best_i != -1:
                new_tracks[tid] = boxes[best_i]
                matched_boxes.add(best_i)

        for i, b in enumerate(boxes):
            if i not in matched_boxes:
                new_tracks[next_id] = b
                next_id += 1

        tracks = new_tracks
        for tid, (x1, y1, x2, y2) in tracks.items():
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(vis_frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(vis_frame, str(tid), (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(vis_frame)
        idx += 1
        cv2.imshow("Tracking",vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stopped by user ('q' pressed).")
            break

    cap.release()
    out.release()

per_frame_boxes = [d["boxes"] for d in all_frames]
track_and_save("match5-c1.avi","outputlab11b.avi", per_frame_boxes)


    
