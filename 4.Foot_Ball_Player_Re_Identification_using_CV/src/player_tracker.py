import cv2
import numpy as np
from collections import defaultdict
import math

class PlayerTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # No detections, mark all as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
            
        # Calculate centroids
        input_centroids = []
        for detection in detections:
            cx = (detection[0] + detection[2]) / 2
            cy = (detection[1] + detection[3]) / 2
            input_centroids.append((cx, cy))
            
        if len(self.objects) == 0:
            # No existing objects, register all
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects with detections
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Calculate distance matrix
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i][j] = self.calculate_distance(obj_centroid, input_centroid)
            
            # Assign using minimum distance
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if distances[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, distances.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, distances.shape[1])).difference(used_col_indices)
            
            # Mark unmatched objects as disappeared
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            for col in unused_col_indices:
                self.register(input_centroids[col])
        
        return self.objects