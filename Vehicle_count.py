import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from strong_sort.strong_sort import StrongSORT
import torch
import csv

cap = cv2.VideoCapture(0)  # For Video

Device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device in use: {Device}")
model = YOLO("weights/yolov8l.pt").to(Device)

classes_path = "coco.names"
with open(classes_path, "r") as f:
    classNames = f.read().strip().split("\n")

# Create a list of random colors to represent each class
np.random.seed(42)  # To get the same colors
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype=np.uint8)

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
required_class_index = [2, 3, 5, 7, 0]

# Dictionary to store the history of object positions
object_history = {}
max_length_of_trail = 20  # Maximum length of the trail
trail_thickness = 4  # Thickness of the trail lines

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0, 0]
down_list = [0, 0, 0, 0, 0]


# Middle cross line position
middle_line_position = 300
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15



def count_vehicle(box_id, iy):

    x, y, w, h, id, index = box_id
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            if 'Up' in up_list:
                up_list.remove('Up')
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            if 'Down' in down_list:
                down_list.remove('Down')
            down_list[index] = down_list[index] + 1
    

    # Draw circle in the middle of the rectangle
    # cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)






while True:
    success, img = cap.read()
    
    ih, iw = img.shape[:2]

    results = model(img, classes=[0, 67], stream=True)

    for result in results:
        boxes = result.boxes
        cls = boxes.cls.tolist()
        conf = boxes.conf
        xywh = boxes.xywh
        # for classIndex in cls:
        #     currentClass = classNames[int(classIndex)]

    pred_cls = np.array(cls)
    conf = conf.detach().cpu().numpy()
    boxes_xywh = xywh.cpu().numpy()
    boxes_xywh = np.array(boxes_xywh, dtype=float)

    resultsTracker = tracker.update(boxes_xywh, conf, pred_cls, img)

    # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Update object history and draw trails
    for result in resultsTracker:
        x1, y1, x2, y2, track_id, class_id, conf = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        if conf > conf_threshold:
            # Draw bounding box
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

            # Put text on image
            cv2.putText(img, f'{int(track_id)}:{classNames[int(class_id)]}:{conf:.2f}',
                        (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Ensure track_id exists in object_history
            object_history.setdefault(track_id, [])
            
            # Draw circle at the center
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Draw trails
            if len(object_history[track_id]) > 1:
                
                # Calculate the number of points to remove from the history
                num_points_to_remove = max(0, len(object_history[track_id]) - max_length_of_trail)
                del object_history[track_id][:num_points_to_remove]  # Remove the oldest points from the history

                for i in range(1, len(object_history[track_id])):
                    # Calculate alpha value based on the index i
                    alpha = int(255 * (1 - i / len(object_history[track_id])))
                    color = tuple(map(int, colors[int(class_id)] + [alpha]))  # Add alpha to the color tuple
        
                    # Draw line from center of current bounding box to center of previous bounding box
                    curr_center = object_history[track_id][i]
                    prev_center = object_history[track_id][i - 1]
                    cv2.line(img, curr_center, prev_center, color, trail_thickness)

            # Add the current point to the object history
            object_history.setdefault(track_id, []).append((cx, cy))
            
            # print(object_history[track_id])
            
            index = required_class_index.index(int(class_id))
            box_id = [x1, y1, w, h, track_id, index]
         
            count_vehicle(box_id, cy)

            # Draw the crossing lines
        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

      
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "person:     "+str(up_list[4])+"     "+ str(down_list[4]), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        
        
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
    # with open("data.csv", 'w') as f1:
    #     cwriter = csv.writer(f1)
    #     cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck','Person'])
    #     up_list.insert(0, "Up")
    #     down_list.insert(0, "Down")
        
    #     cwriter.writerow(up_list)
    #     cwriter.writerow(down_list)
    # f1.close()

cap.release()
cv2.destroyAllWindows()
