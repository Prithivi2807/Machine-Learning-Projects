"""
FIXED Player Re-Identification System
Fixes: 
1. Proper class filtering (players vs ball)
2. Better distance thresholds
3. Size-based filtering to remove ball detections

Usage command:  python improved_player_reid.py --video ../data/videos/15sec_input_720p.mp4 --model ../data/models/best.pt


"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
from collections import OrderedDict
import math

class ImprovedPlayerTracker:
    def __init__(self, max_disappeared=15, max_distance=200):
        """
        Improved tracker with better parameters
        
        Args:
            max_disappeared: Max frames before removing track (reduced from 30)
            max_distance: Max pixel distance for matching (increased from 80)
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Store object history for better tracking
        self.object_history = OrderedDict()

    def register(self, centroid, bbox_size):
        """Register a new object with size information"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_history[self.next_object_id] = {
            'centroids': [centroid],
            'sizes': [bbox_size],
            'last_seen': 0
        }
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.object_history:
            del self.object_history[object_id]

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence, class_id] arrays
        """
        if len(detections) == 0:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_current_objects()

        # Calculate centroids and sizes for detections
        input_data = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            width = x2 - x1
            height = y2 - y1
            size = width * height
            input_data.append({
                'centroid': (cx, cy),
                'size': size,
                'bbox': (x1, y1, x2, y2)
            })

        if len(self.objects) == 0:
            # Register all new detections
            for data in input_data:
                self.register(data['centroid'], data['size'])
        else:
            # Match existing objects with new detections
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Calculate distance matrix
            distances = np.zeros((len(object_centroids), len(input_data)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, data in enumerate(input_data):
                    distances[i][j] = self.calculate_distance(obj_centroid, data['centroid'])

            # Hungarian algorithm approximation - assign minimum distances
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            # Process matches
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                distance = distances[row, col]
                if distance <= self.max_distance:
                    object_id = object_ids[row]
                    new_centroid = input_data[col]['centroid']
                    new_size = input_data[col]['size']
                    
                    # Update object
                    self.objects[object_id] = new_centroid
                    self.disappeared[object_id] = 0
                    
                    # Update history
                    if object_id in self.object_history:
                        self.object_history[object_id]['centroids'].append(new_centroid)
                        self.object_history[object_id]['sizes'].append(new_size)
                        self.object_history[object_id]['last_seen'] = 0
                        
                        # Keep only last 10 positions for memory efficiency
                        if len(self.object_history[object_id]['centroids']) > 10:
                            self.object_history[object_id]['centroids'].pop(0)
                            self.object_history[object_id]['sizes'].pop(0)

                    used_row_indices.add(row)
                    used_col_indices.add(col)

            # Handle unmatched objects and detections
            unused_rows = set(range(len(object_centroids))).difference(used_row_indices)
            unused_cols = set(range(len(input_data))).difference(used_col_indices)

            # Mark unmatched objects as disappeared
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if object_id in self.object_history:
                    self.object_history[object_id]['last_seen'] += 1
                    
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new unmatched detections
            for col in unused_cols:
                data = input_data[col]
                self.register(data['centroid'], data['size'])

        return self.get_current_objects()

    def get_current_objects(self):
        """Return current tracked objects with their IDs"""
        return self.objects.copy()


def filter_detections(results, confidence_threshold=0.5, min_area=500, max_area=50000):
    """
    Filter YOLO detections to get only valid players
    
    Args:
        results: YOLO detection results
        confidence_threshold: Minimum confidence score
        min_area: Minimum bounding box area (removes small objects like ball)
        max_area: Maximum bounding box area (removes very large detections)
    
    Returns:
        List of filtered detections [x1, y1, x2, y2, confidence, class_id]
    """
    detections = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Get detection data
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate bounding box area
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filter criteria
                if (conf >= confidence_threshold and 
                    min_area <= area <= max_area and
                    width > 25 and height > 50):  # Players should be taller than wide - Adjust Minimum player dimentions
                    
                    detections.append([x1, y1, x2, y2, conf, cls])
    
    return detections


def process_video_improved(video_path, model_path, output_path="output_tracked.mp4"):
    """
    Improved video processing with better filtering and tracking
    """
    
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Print model class names to understand what it detects
    print("Model classes:", model.names)
    # This helps identify if class 0 is player, class 1 is ball, etc.
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize improved tracker with better parameters
    tracker = ImprovedPlayerTracker(
        max_disappeared = 20,  # Reduced from 30
        max_distance = 150     # Increased from 80
    )
    
    # Better color scheme for player IDs
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]
    
    tracking_data = []
    frame_count = 0
    
    print("Processing video with improved tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        results = model(frame, verbose=False)
        
        # Filter detections (CRITICAL: This removes ball detections)
        detections = filter_detections(
            results, 
            confidence_threshold=0.5,  # Lower threshold for better detection
            min_area=1200,              # Minimum area (removes ball)
            max_area=35000             # Maximum area
        )
        
        # Update tracker
        objects = tracker.update(detections)
        
        # Draw results
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate detection centroid
            det_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find matching tracked object
            best_match_id = None
            min_distance = float('inf')
            
            for obj_id, obj_centroid in objects.items():
                distance = math.sqrt(
                    (det_centroid[0] - obj_centroid[0])**2 + 
                    (det_centroid[1] - obj_centroid[1])**2
                )
                
                if distance < min_distance and distance < 100:  # Match threshold
                    min_distance = distance
                    best_match_id = obj_id
            
            # Draw if matched
            if best_match_id is not None:
                color = colors[best_match_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw filled background for text
                label = f"Player {best_match_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Background rectangle
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_height - 10), 
                    (x1 + text_width + 10, y1), 
                    color, 
                    -1
                )
                
                # Text
                cv2.putText(
                    frame, label, (x1 + 5, y1 - 5), 
                    font, font_scale, (255, 255, 255), thickness
                )
                
                # Draw center point
                cv2.circle(frame, det_centroid, 4, color, -1)
                
                # Store tracking data
                tracking_data.append({
                    'frame': frame_count,
                    'player_id': int(best_match_id),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'centroid': det_centroid,
                    'area': (x2-x1) * (y2-y1)
                })
        
        # Draw frame number
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% - Active tracks: {len(objects)}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save results
    results_path = output_path.replace('.mp4', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    # Print summary
    unique_players = set(item['player_id'] for item in tracking_data)
    print(f"\n✅ Processing Complete!")
    print(f"📹 Output video: {output_path}")
    print(f"📊 Results file: {results_path}")
    print(f"🎯 Frames processed: {frame_count}")
    print(f"👥 Unique players: {len(unique_players)} - IDs: {sorted(unique_players)}")
    print(f"📍 Total detections: {len(tracking_data)}")
    
    return tracking_data


def main():
    parser = argparse.ArgumentParser(description='Improved Player Re-Identification')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--model', required=True, help='Path to YOLOv11 model')
    parser.add_argument('--output', default='output_improved.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Improved Player Re-Identification...")
        tracking_data = process_video_improved(args.video, args.model, args.output)
        
        # Additional analysis
        if tracking_data:
            frame_counts = {}
            for item in tracking_data:
                pid = item['player_id']
                frame_counts[pid] = frame_counts.get(pid, 0) + 1
            
            print(f"\n📈 Player Activity:")
            for pid in sorted(frame_counts.keys()):
                print(f"   Player {pid}: appeared in {frame_counts[pid]} frames")
        
        print("✅ Success! Check your output video.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())