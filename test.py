import torch
from matplotlib import pyplot as plt
import cv2

# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_v5.pt')

cap = cv2.VideoCapture("note.mp4")

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model(frame)
        results.show()
        #cv2.imshow("res", results.render()[0])
        #print(results)
        #cv2.waitKey(1)
print("END")
