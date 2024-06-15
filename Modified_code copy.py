import numpy as np
from ultralytics import YOLO
import cv2
import math
from strong_sort.strong_sort import StrongSORT
import torch
import csv
import time
import os
import logging
from collections import deque
import queue
import threading

class VehicleCounter:
    def __init__(self, required_class_index, class_names):
        self.up_list = [0] * len(required_class_index)
        self.down_list = [0] * len(required_class_index)
        self.temp_list = {}
        self.copy_up_list = [0] * len(required_class_index)
        self.copy_down_list = [0] * len(required_class_index)
        self.required_class_index = required_class_index
        self.required_class_names = [class_names[i] for i in required_class_index]

        # New data structures
        self.track_id_last_seen = {}
        self.up_counted_ids = set()
        self.down_counted_ids = set()

    def count_vehicle(self, box_id, cx, cy, line_start, line_end):
        x, y, w, h, id, index = box_id
        current_time = time.time()

        # Update last seen timestamp for the track ID
        self.track_id_last_seen[id] = current_time

        # Compute the cross product to determine which side of the line the point is on
        current_side = (line_end[0] - line_start[0]) * (cy - line_start[1]) - (line_end[1] - line_start[1]) * (cx - line_start[0])

        if id not in self.temp_list:
            self.temp_list[id] = current_side

        if self.temp_list[id] * current_side < 0:
            if self.temp_list[id] > 0:
                if id not in self.down_counted_ids:
                    self.down_list[index] += 1
                    self.down_counted_ids.add(id)
                    print(f"Vehicle {id} counted as DOWN at {cx}, {cy}")
            else:
                if id not in self.up_counted_ids:
                    self.up_list[index] += 1
                    self.up_counted_ids.add(id)
                    print(f"Vehicle {id} counted as UP at {cx}, {cy}")
            del self.temp_list[id]

    def get_counts(self):
        result_up = [self.up_list[i] - self.copy_up_list[i] for i in range(len(self.up_list))]
        result_down = [self.down_list[i] - self.copy_down_list[i] for i in range(len(self.down_list))]
        self.copy_up_list = self.up_list.copy()
        self.copy_down_list = self.down_list.copy()
        return result_up, result_down

    def reset(self):
        self.up_list = [0] * len(self.required_class_index)
        self.down_list = [0] * len(self.required_class_index)
        self.temp_list = {}

    def cleanup_track_ids(self):
        current_time = time.time()
        ids_to_remove = [track_id for track_id, last_seen in self.track_id_last_seen.items() if current_time - last_seen > 2]

        for track_id in ids_to_remove:
            del self.track_id_last_seen[track_id]
            self.up_counted_ids.discard(track_id)
            self.down_counted_ids.discard(track_id)


def initialize_csv(output_dir, required_class_names):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_datetime = time.strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(output_dir, f'vehicle_counts_{current_datetime}.csv')
    csv_fieldnames = ['Time', 'Direction'] + required_class_names

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fieldnames)
        writer.writeheader()
    
    return csv_filename


def save_counts_to_csv(csv_filename, counts_up, counts_down, required_class_names):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Time', 'Direction'] + required_class_names)
        writer.writerow({'Time': time.strftime('%Y-%m-%d %H:%M:%S'), 'Direction': 'Up', **dict(zip(required_class_names, counts_up))})
        writer.writerow({'Time': time.strftime('%Y-%m-%d %H:%M:%S'), 'Direction': 'Down', **dict(zip(required_class_names, counts_down))})


def draw_bounding_box_and_trail(img, x1, y1, x2, y2, track_id, class_id, conf, colors, class_names, object_history, max_length_of_trail, trail_thickness):
    w, h = x2 - x1, y2 - y1
    cx, cy = int(x1 + w / 2), int(y1 + h / 2)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    # Draw label text
    cv2.putText(img, f'{int(track_id)}:{class_names[int(class_id)]}:{conf:.2f}', 
                (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    # Draw center point
    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # Initialize history for the track_id if not already present
    if track_id not in object_history:
        object_history[track_id] = deque(maxlen=max_length_of_trail)
    
    # Add the current center to the history
    if not object_history[track_id] or object_history[track_id][-1] != (cx, cy):
        object_history[track_id].append((cx, cy))
    
    # Draw the trail
    for i in range(1, len(object_history[track_id])):
        alpha = int(255 * (1 - i / len(object_history[track_id])))
        color = (255, 0, 255, alpha)
        curr_center = object_history[track_id][i]
        prev_center = object_history[track_id][i - 1]
        cv2.line(img, curr_center, prev_center, (255, 0, 255), trail_thickness)
    
    return cx, cy


def open_rtsp_stream(rtsp_url, retries=5, delay=5):
    for i in range(retries):
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            logging.info("RTSP stream opened successfully.")
            return cap
        else:
            logging.warning(f"Failed to open RTSP stream. Retrying {i + 1}/{retries}...")
            time.sleep(delay)
    logging.error("Failed to open RTSP stream after multiple attempts.")
    return None


def smooth_boxes(boxes, alpha=0.5):
    """
    Smooth the bounding boxes using an exponential moving average.
    boxes: List of bounding boxes [x1, y1, x2, y2]
    alpha: Smoothing factor between 0 and 1
    """
    if not hasattr(smooth_boxes, 'previous_boxes'):
        smooth_boxes.previous_boxes = boxes
    smoothed_boxes = alpha * np.array(boxes) + (1 - alpha) * np.array(smooth_boxes.previous_boxes)
    smooth_boxes.previous_boxes = smoothed_boxes
    return smoothed_boxes

def buffered_video_capture(rtsp_url, buffer_size=2, retries=5, delay=5):
    cap = open_rtsp_stream(rtsp_url, retries, delay)
    if cap is None:
        return None, None, None

    q = queue.Queue(maxsize=buffer_size)
    stop_event = threading.Event()

    def capture_loop():
        nonlocal cap
        while not stop_event.is_set():
            if not q.full():
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Frame capture failed. Attempting to reopen RTSP stream.")
                    cap.release()
                    cap = open_rtsp_stream(rtsp_url, retries, delay)
                    if cap is None:
                        logging.error("Failed to reopen RTSP stream. Will keep trying.")
                        while cap is None and not stop_event.is_set():
                            time.sleep(delay)
                            cap = open_rtsp_stream(rtsp_url, retries, delay)
                else:
                    q.put(frame)
            else:
                time.sleep(0.01)

    thread = threading.Thread(target=capture_loop)
    thread.daemon = True
    thread.start()

    return q, stop_event, thread


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # rtsp_url = "rtsp://admin:admin@123@192.168.50.101:554/cam/realmonitor?channel=1&subtype=1"
    rtsp_url = 0
    buffer_size = 2

    q, stop_event, capture_thread = buffered_video_capture(rtsp_url, buffer_size=buffer_size)

    if q is None:
        logging.error("Exiting due to failure to open RTSP stream.")
        return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Device in use: {device}")
    model = YOLO("weights/yolov8n.pt").to(device)

    classes_path = "coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    tracker = StrongSORT(model_weights='weights/osnet_x0_25_msmt17.pt', device=device,
                         max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, mc_lambda=0.995, ema_alpha=0.9)

    conf_threshold = 0.5
    required_class_index = [2, 3, 5, 7, 0]

    vehicle_counter = VehicleCounter(required_class_index, class_names)

    object_history = {}
    max_length_of_trail = 10
    trail_thickness = 4

    font_color = (0, 0, 255)
    font_size = 0.5
    font_thickness = 2

    # Define the line and center point
    line_start = (0, 300)
    line_end = (640, 300)

    output_dir = 'Output_csv'
    csv_filename = initialize_csv(output_dir, vehicle_counter.required_class_names)

    start_time = time.time()
    save_interval = 10

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= save_interval:
            counts_up, counts_down = vehicle_counter.get_counts()
            save_counts_to_csv(csv_filename, counts_up, counts_down, vehicle_counter.required_class_names)
            start_time = time.time()

        if not q.empty():
            img = q.get()
        else:
            time.sleep(0.01)
            continue

        ih, iw = img.shape[:2]
        results = model(img, classes=required_class_index, stream=True)

        for result in results:
            boxes = result.boxes
            cls = boxes.cls.tolist()
            conf = boxes.conf
            xywh = boxes.xywh

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        boxes_xywh = xywh.cpu().numpy()
        boxes_xywh = np.array(boxes_xywh, dtype=float)

        results_tracker = tracker.update(boxes_xywh, conf, pred_cls, img)

        for result in results_tracker:
            x1, y1, x2, y2, track_id, class_id, conf = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if conf > conf_threshold:
                cx, cy = draw_bounding_box_and_trail(img, x1, y1, x2, y2, track_id, class_id, conf, colors, class_names, object_history, max_length_of_trail, trail_thickness)
                index = required_class_index.index(int(class_id))
                box_id = [x1, y1, x2 - x1, y2 - y1, track_id, index]
                vehicle_counter.count_vehicle(box_id, cx, cy, line_start, line_end)

        cv2.line(img, line_start, line_end, (255, 0, 255), 4)

        # Display the counts on the image
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(vehicle_counter.up_list[0])+"     "+ str(vehicle_counter.down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(vehicle_counter.up_list[1])+"     "+ str(vehicle_counter.down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(vehicle_counter.up_list[2])+"     "+ str(vehicle_counter.down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(vehicle_counter.up_list[3])+"     "+ str(vehicle_counter.down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "person:     "+str(vehicle_counter.up_list[4])+"     "+ str(vehicle_counter.down_list[4]), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

        # Clean up old track IDs
        vehicle_counter.cleanup_track_ids()

    stop_event.set()
    capture_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
