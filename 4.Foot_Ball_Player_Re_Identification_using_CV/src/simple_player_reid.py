"""
Simple Player Re-Identification System
Based on Centroid Tracking

Usage:  python src/simple_player_reid.py --video data/videos/15sec_input_720p.mp4 --model data/models/best.pt
"""
import os 
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
from collections import OrderedDict
import math


print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
class SimplePlayerTracker:
    def __init__(self, max_disappeared=20, max_distance=100):
        """
        Simple centroid-based tracker for player re-identification
        
        Args:
            max_disappeared: Max frames an object can be missing before removal
            max_distance: Max distance for matching detections to existing tracks
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Register a new object with the next available ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update the tracker with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence] arrays
            
        Returns:
            Dictionary of {object_id: (centroid_x, centroid_y)}
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_current_objects()

        # Calculate centroids for new detections
        input_centroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))

        # If no existing objects, register all new detections
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects with new detections
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Calculate distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                              np.array(input_centroids), axis=2)

            # Find the minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            # Process matches
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0

                    used_row_indices.add(row)
                    used_col_indices.add(col)

            # Handle unmatched detections and existing objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            # Mark unmatched existing objects as disappeared
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new unmatched detections
            for col in unused_col_indices:
                self.register(input_centroids[col])

        return self.get_current_objects()

    def get_current_objects(self):
        """Return current tracked objects"""
        return self.objects.copy()


def process_video(video_path, model_path, output_path="output_tracked.mp4", 
                 confidence_threshold=0.5):
    """
    Process video for player re-identification
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLOv11 model
        output_path: Path for output video
        confidence_threshold: Minimum confidence for detections
    """
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = SimplePlayerTracker(max_disappeared=30, max_distance=80)
    
    # Colors for different player IDs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)
    ]
    
    tracking_data = []
    frame_count = 0
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Extract detections (assuming class 0 is 'person' or 'player')
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Filter by confidence and class (assuming person/player is class 0)
                    if box.conf[0] >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf])
        
        # Update tracker
        objects = tracker.update(detections)
        
        # Draw detections and tracking results
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Find corresponding tracked object
            detection_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find closest tracked object
            closest_id = None
            min_distance = float('inf')
            
            for obj_id, obj_centroid in objects.items():
                distance = math.sqrt((detection_centroid[0] - obj_centroid[0])**2 + 
                                   (detection_centroid[1] - obj_centroid[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_id = obj_id
            
            # Draw bounding box and ID if matched
            if closest_id is not None and min_distance < 60:
                color = colors[closest_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw player ID
                label = f"Player {closest_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Store tracking data
                tracking_data.append({
                    'frame': frame_count,
                    'player_id': int(closest_id),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'centroid': detection_centroid
                })
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save tracking results
    results_path = output_path.replace('.mp4', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Output video: {output_path}")
    print(f"Tracking results: {results_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total tracked detections: {len(tracking_data)}")
    
    return tracking_data


def main():
    parser = argparse.ArgumentParser(description='Simple Player Re-Identification')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--model', required=True, help='Path to YOLOv11 model')
    parser.add_argument('--output', default='output_tracked.mp4', help='Output video path')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    try:
        tracking_data = process_video(args.video, args.model, args.output, args.confidence)
        
        # Print summary statistics
        unique_players = set(item['player_id'] for item in tracking_data)
        print(f"\nSummary:")
        print(f"Unique playerbs tracked: {len(unique_players)}")
        print(f"Player IDs: {sorted(unique_players)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())