import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from strong_sort.strong_sort import StrongSORT
import torch

# print(torch.cuda.is_available())

cap = cv2.VideoCapture(0)  # For Video

Device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device in use: {Device}")
model = YOLO("weights/yolov8l.pt").to(Device)

classes_path = "coco.names"
with open(classes_path, "r") as f:
    classNames = f.read().strip().split("\n")
    
# create a list of random colors to represent each class
np.random.seed(42)  # to get the same colors
colors = np.random.randint(0, 255, size=(len(classNames), 3))

# Tracking
tracker = StrongSORT(model_weights='weights/osnet_x0_25_msmt17.pt', device=Device,
                    max_dist=0.2,
                    max_iou_distance=0.7,
                    max_age=70,
                    n_init=3,
                    nn_budget=100,
                    mc_lambda=0.995,
                    ema_alpha=0.9)

limits = [0, 250, 673, 250]
totalCount = []
conf_threshold = 0.5


while True:
    success, img = cap.read()
    bbox_list = []
    class_list = []
    conf_list = []

    results = model(img, classes=[0, 67], stream=True)

    for result in results:
    
        boxes = result.boxes
        cls = boxes.cls.tolist()
        # xyxy = boxes.xyxy
        conf = boxes.conf
        xywh = boxes.xywh
        for classIndex in cls:
            currentClass = classNames[int(classIndex)]   
        
        
    pred_cls = np.array(cls)
    conf = conf.detach().cpu().numpy()
    # xyxy = xyxy.detach().cpu().numpy()
    boxes_xywh = xywh.cpu().numpy()
    boxes_xywh = np.array(boxes_xywh, dtype = float)
        

    resultsTracker = tracker.update(boxes_xywh, conf, pred_cls, img)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, track_id, class_id, conf = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)                     
        w, h = x2 - x1, y2 - y1

        if conf > conf_threshold:
            # Draw bounding box
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

            # Put text on image
            cv2.putText(img, f'ID: {int(track_id)} | Class: {classNames[int(class_id)]} | Confidence: {conf:.2f}',
                    (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            # Draw circle at the center
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check if the center is within limits and draw appropriate line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(track_id) == 0:
                    totalCount.append(track_id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

