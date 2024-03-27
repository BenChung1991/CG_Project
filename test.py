from ultralytics import YOLO
import cv2
 
model = YOLO('best.pt')
results = model("t1.png", show=True)

#results.names, results.scores, results.xyxy
cv2.waitKey(0)
