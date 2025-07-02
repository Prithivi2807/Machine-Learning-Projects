import cv2
from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_players(self, frame):
        """Detect players in a single frame"""
        results = self.model(frame)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls == 0:  # Assuming class 0 is player
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections)