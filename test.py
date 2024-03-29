from ultralytics import YOLO
import cv2
 
model = YOLO('bestm2.pt')
results = model("234.jpg", show=True)

#results.names, results.scores, results.xyxy
cv2.waitKey(0)
