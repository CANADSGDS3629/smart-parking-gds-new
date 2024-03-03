import cv2
import numpy as np
import cvzone
import pickle
import pandas as pd
from ultralytics import YOLO

area_names = []
polylines = []

# Function to draw regions of interest (ROIs)
def draw(event, x, y, flags, param):
    global points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_name = input('Enter area name: ')
        if current_name:
            area_names.append(current_name)
            polylines.append(np.array(points, np.int32))

cap = cv2.VideoCapture('easy1.mp4')

drawing = False
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1020, 500))

    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], isClosed=True, color=(0, 0, 255), thickness=2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()  # Close window and then break the loop
        break
    if key == ord('s'):
        with open("freedomtech", "wb") as f:
            data = {'polylines': polylines, 'area_names': area_names}
            pickle.dump(data, f)

cap.release()


# Now, executing the logic and code from test2.py

# Loading existing data
with open("freedomtech", "rb") as f:
    data = pickle.load(f)
    polylines, area_names = data['polylines'], data['area_names']

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture('easy1.mp4')

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()

    # Object detection
    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    # Extracting car positions
    car_positions = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, d = row[:6]
        c = class_list[int(d)]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 'car' in c:
            car_positions.append([cx, cy])

    # Counting cars in each area
    car_counter = []
    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], isClosed=True, color=(0, 255, 0), thickness=2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)
        for cx1, cy1 in car_positions:
            result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
            if result >= 0:
                cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [polyline], isClosed=True, color=(0, 0, 255), thickness=2)
                car_counter.append(cx1)

    # Displaying car count and free space count
    car_count = len(car_counter)
    free_space = len(polylines) - car_count
    cvzone.putTextRect(frame, f'CARCOUNTER: {car_count}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'freespace: {free_space}', (50, 160), 2, 2)

    cv2.imshow('FRAME', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()