"""
ENHANCED Player Re-Identification System with Persistent Tracking
Improvements:
1. Multi-feature matching (position, size, appearance region)
2. Longer disappearance tolerance with dormant state
3. Re-identification logic for returning players
4. Better trajectory prediction
5. Confidence-based ID assignment

Usage: python enhanced_player_reid_01.py --video ../data/videos/15sec_input_720p.mp4 --model ../data/models/best.pt
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import json
from collections import OrderedDict, deque
import math
import time

class EnhancedPlayerTracker:
    def __init__(self, max_disappeared=30, max_dormant=150, max_distance=120):
        """
        Enhanced tracker with persistent re-identification
        
        Args:
            max_disappeared: Frames before moving to dormant state
            max_dormant: Total frames before permanent removal
            max_distance: Max pixel distance for matching
        """
        self.next_object_id = 0
        self.active_objects = OrderedDict()      # Currently visible objects
        self.dormant_objects = OrderedDict()     # Recently disappeared objects
        self.disappeared = OrderedDict()
        self.dormant_timer = OrderedDict()
        
        self.max_disappeared = max_disappeared
        self.max_dormant = max_dormant
        self.max_distance = max_distance
        
        # Enhanced tracking data
        self.object_profiles = OrderedDict()  # Detailed object information
        self.frame_count = 0

    def create_object_profile(self, centroid, bbox, appearance_roi=None):
        """Create comprehensive object profile for re-identification"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        profile = {
            'creation_frame': self.frame_count,
            'last_seen_frame': self.frame_count,
            'centroids': deque([centroid], maxlen=20),
            'bboxes': deque([bbox], maxlen=20),
            'sizes': deque([(width, height)], maxlen=20),
            'areas': deque([width * height], maxlen=20),
            'velocities': deque(maxlen=10),
            'avg_size': (width, height),
            'avg_area': width * height,
            'position_variance': 0.0,
            'total_detections': 1,
            'confidence_history': deque(maxlen=10),
            'appearance_roi': appearance_roi  # Small image patch for appearance matching
        }
        return profile

    def update_object_profile(self, object_id, centroid, bbox, confidence=None, appearance_roi=None):
        """Update existing object profile with new detection"""
        if object_id not in self.object_profiles:
            return
            
        profile = self.object_profiles[object_id]
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        
        # Update position history
        if len(profile['centroids']) > 0:
            last_centroid = profile['centroids'][-1]
            velocity = (centroid[0] - last_centroid[0], centroid[1] - last_centroid[1])
            profile['velocities'].append(velocity)
        
        profile['centroids'].append(centroid)
        profile['bboxes'].append(bbox)
        profile['sizes'].append((width, height))
        profile['areas'].append(width * height)
        profile['last_seen_frame'] = self.frame_count
        profile['total_detections'] += 1
        
        if confidence is not None:
            profile['confidence_history'].append(confidence)
        
        if appearance_roi is not None:
            profile['appearance_roi'] = appearance_roi
        
        # Update averages
        recent_areas = list(profile['areas'])[-5:]  # Last 5 detections
        recent_sizes = list(profile['sizes'])[-5:]
        
        profile['avg_area'] = sum(recent_areas) / len(recent_areas)
        profile['avg_size'] = (
            sum(s[0] for s in recent_sizes) / len(recent_sizes),
            sum(s[1] for s in recent_sizes) / len(recent_sizes)
        )
        
        # Calculate position variance (stability measure)
        if len(profile['centroids']) >= 3:
            recent_centroids = list(profile['centroids'])[-10:]
            x_coords = [c[0] for c in recent_centroids]
            y_coords = [c[1] for c in recent_centroids]
            profile['position_variance'] = np.var(x_coords) + np.var(y_coords)

    def predict_position(self, object_id):
        """Predict next position based on velocity"""
        if object_id not in self.object_profiles:
            return None
            
        profile = self.object_profiles[object_id]
        
        if len(profile['centroids']) < 2:
            return profile['centroids'][-1] if profile['centroids'] else None
        
        # Use recent velocities for prediction
        if len(profile['velocities']) > 0:
            recent_vels = list(profile['velocities'])[-3:]  # Last 3 velocities
            avg_vel_x = sum(v[0] for v in recent_vels) / len(recent_vels)
            avg_vel_y = sum(v[1] for v in recent_vels) / len(recent_vels)
            
            last_pos = profile['centroids'][-1]
            predicted_pos = (
                int(last_pos[0] + avg_vel_x),
                int(last_pos[1] + avg_vel_y)
            )
            return predicted_pos
        
        return profile['centroids'][-1]

    def calculate_matching_score(self, detection_data, object_id, use_prediction=True):
        """
        Calculate comprehensive matching score between detection and tracked object
        Lower score = better match
        """
        if object_id not in self.object_profiles:
            return float('inf')
        
        profile = self.object_profiles[object_id]
        det_centroid = detection_data['centroid']
        det_size = detection_data['size']
        det_area = detection_data['area']
        
        # 1. Position distance (with prediction)
        if use_prediction:
            predicted_pos = self.predict_position(object_id)
            if predicted_pos:
                pos_distance = math.sqrt(
                    (det_centroid[0] - predicted_pos[0])**2 + 
                    (det_centroid[1] - predicted_pos[1])**2
                )
            else:
                pos_distance = float('inf')
        else:
            if object_id in self.active_objects:
                last_pos = self.active_objects[object_id]
            elif object_id in self.dormant_objects:
                last_pos = self.dormant_objects[object_id]
            else:
                return float('inf')
                
            pos_distance = math.sqrt(
                (det_centroid[0] - last_pos[0])**2 + 
                (det_centroid[1] - last_pos[1])**2
            )
        
        # 2. Size consistency
        avg_area = profile['avg_area']
        area_ratio = min(det_area, avg_area) / max(det_area, avg_area)
        size_score = (1.0 - area_ratio) * 100  # Convert to penalty
        
        # 3. Aspect ratio consistency
        det_width, det_height = det_size
        det_ratio = det_width / max(det_height, 1)
        avg_width, avg_height = profile['avg_size']
        avg_ratio = avg_width / max(avg_height, 1)
        ratio_diff = abs(det_ratio - avg_ratio)
        ratio_score = ratio_diff * 50
        
        # 4. Trajectory consistency
        trajectory_score = 0
        if len(profile['velocities']) > 2:
            recent_vels = list(profile['velocities'])[-3:]
            avg_speed = np.mean([math.sqrt(v[0]**2 + v[1]**2) for v in recent_vels])
            if avg_speed > 50:  # Fast moving object
                trajectory_score = min(pos_distance * 0.5, 50)
        
        # 5. Time penalty for dormant objects
        time_penalty = 0
        if object_id in self.dormant_timer:
            frames_dormant = self.dormant_timer[object_id]
            time_penalty = frames_dormant * 2  # Increase penalty over time
        
        # Combined score (weighted)
        total_score = (
            pos_distance * 1.0 +      # Position weight
            size_score * 0.3 +        # Size weight  
            ratio_score * 0.2 +       # Aspect ratio weight
            trajectory_score * 0.1 +  # Trajectory weight
            time_penalty * 0.4        # Time penalty weight
        )
        
        return total_score

    def register_new_object(self, detection_data, frame=None):
        """Register a completely new object"""
        object_id = self.next_object_id
        centroid = detection_data['centroid']
        bbox = detection_data['bbox']
        
        # Extract appearance ROI if frame provided
        appearance_roi = None
        if frame is not None:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            appearance_roi = frame[y1:y2, x1:x2].copy()
        
        # Create profile
        self.object_profiles[object_id] = self.create_object_profile(
            centroid, bbox, appearance_roi
        )
        
        # Add to active tracking
        self.active_objects[object_id] = centroid
        self.disappeared[object_id] = 0
        
        self.next_object_id += 1
        return object_id

    def reactivate_dormant_object(self, object_id, detection_data, frame=None):
        """Reactivate a dormant object that has reappeared"""
        centroid = detection_data['centroid']
        bbox = detection_data['bbox']
        
        # Move from dormant to active
        if object_id in self.dormant_objects:
            del self.dormant_objects[object_id]
        if object_id in self.dormant_timer:
            del self.dormant_timer[object_id]
        
        self.active_objects[object_id] = centroid
        self.disappeared[object_id] = 0
        
        # Update profile
        confidence = detection_data.get('confidence', None)
        appearance_roi = None
        if frame is not None:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            appearance_roi = frame[y1:y2, x1:x2].copy()
        
        self.update_object_profile(object_id, centroid, bbox, confidence, appearance_roi)
        
        return object_id

    def update(self, detections, frame=None):
        """
        Enhanced update with re-identification capabilities
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            self._handle_no_detections()
            return self.get_current_objects()

        # Prepare detection data
        detection_data = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 1.0
            
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            width, height = x2 - x1, y2 - y1
            area = width * height
            
            detection_data.append({
                'centroid': centroid,
                'bbox': (x1, y1, x2, y2),
                'size': (width, height),
                'area': area,
                'confidence': confidence
            })

        # Match detections with existing objects
        self._match_detections(detection_data, frame)
        
        return self.get_current_objects()

    def _match_detections(self, detection_data, frame):
        """Match detections to existing objects with re-identification"""
        all_object_ids = list(self.active_objects.keys()) + list(self.dormant_objects.keys())
        
        if len(all_object_ids) == 0:
            # No existing objects, register all as new
            for data in detection_data:
                self.register_new_object(data, frame)
            return

        # Calculate matching scores for all combinations
        match_scores = {}
        for i, data in enumerate(detection_data):
            for object_id in all_object_ids:
                score = self.calculate_matching_score(data, object_id)
                match_scores[(i, object_id)] = score

        # Hungarian-style assignment (greedy approximation)
        used_detections = set()
        used_objects = set()
        matches = []

        # Sort by score and assign greedily
        sorted_matches = sorted(match_scores.items(), key=lambda x: x[1])
        
        for (det_idx, obj_id), score in sorted_matches:
            if det_idx in used_detections or obj_id in used_objects:
                continue
            
            # Apply stricter thresholds based on object state
            if obj_id in self.active_objects:
                threshold = self.max_distance
            else:  # Dormant object
                threshold = self.max_distance * 1.5  # More lenient for dormant objects
            
            if score <= threshold:
                matches.append((det_idx, obj_id, score))
                used_detections.add(det_idx)
                used_objects.add(obj_id)

        # Apply matches
        for det_idx, obj_id, score in matches:
            data = detection_data[det_idx]
            
            if obj_id in self.active_objects:
                # Update active object
                self.active_objects[obj_id] = data['centroid']
                self.disappeared[obj_id] = 0
            else:
                # Reactivate dormant object
                self.reactivate_dormant_object(obj_id, data, frame)
            
            # Update profile
            self.update_object_profile(
                obj_id, data['centroid'], data['bbox'], 
                data['confidence'], None
            )

        # Handle unmatched detections (create new objects)
        for i, data in enumerate(detection_data):
            if i not in used_detections:
                self.register_new_object(data, frame)

        # Handle unmatched objects (mark as disappeared)
        for obj_id in all_object_ids:
            if obj_id not in used_objects:
                if obj_id in self.active_objects:
                    self.disappeared[obj_id] += 1
                    
                    if self.disappeared[obj_id] > self.max_disappeared:
                        # Move to dormant state
                        self.dormant_objects[obj_id] = self.active_objects[obj_id]
                        self.dormant_timer[obj_id] = 0
                        del self.active_objects[obj_id]
                        del self.disappeared[obj_id]
                
                elif obj_id in self.dormant_timer:
                    self.dormant_timer[obj_id] += 1

        # Clean up objects that have been dormant too long
        dormant_to_remove = []
        for obj_id, timer in self.dormant_timer.items():
            if timer > self.max_dormant:
                dormant_to_remove.append(obj_id)

        for obj_id in dormant_to_remove:
            self.permanently_remove_object(obj_id)

    def _handle_no_detections(self):
        """Handle frame with no detections"""
        # Mark all active objects as disappeared
        for object_id in list(self.active_objects.keys()):
            self.disappeared[object_id] += 1
            
            if self.disappeared[object_id] > self.max_disappeared:
                # Move to dormant
                self.dormant_objects[object_id] = self.active_objects[object_id]
                self.dormant_timer[object_id] = 0
                del self.active_objects[object_id]
                del self.disappeared[object_id]

        # Update dormant timers
        for object_id in list(self.dormant_timer.keys()):
            self.dormant_timer[object_id] += 1
            if self.dormant_timer[object_id] > self.max_dormant:
                self.permanently_remove_object(object_id)

    def permanently_remove_object(self, object_id):
        """Permanently remove an object from all tracking"""
        if object_id in self.active_objects:
            del self.active_objects[object_id]
        if object_id in self.dormant_objects:
            del self.dormant_objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.dormant_timer:
            del self.dormant_timer[object_id]
        if object_id in self.object_profiles:
            del self.object_profiles[object_id]

    def get_current_objects(self):
        """Return currently active objects"""
        return self.active_objects.copy()

    def get_all_objects(self):
        """Return both active and dormant objects for debugging"""
        return {
            'active': self.active_objects.copy(),
            'dormant': self.dormant_objects.copy(),
            'dormant_timers': self.dormant_timer.copy()
        }


def filter_detections(results, confidence_threshold=0.4, min_area=800, max_area=40000):
    """Enhanced detection filtering"""
    detections = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / max(height, 1)
                
                # Enhanced filtering criteria
                if (conf >= confidence_threshold and 
                    min_area <= area <= max_area and
                    width > 20 and height > 40 and  # Minimum player dimensions
                    aspect_ratio < 2.0 and  # Players shouldn't be too wide
                    height > width * 0.8):  # Players should be roughly vertical
                    
                    detections.append([x1, y1, x2, y2, conf, cls])
    
    return detections


def process_video_enhanced(video_path, model_path, output_path="output_enhanced.mp4"):
    """Enhanced video processing with persistent player tracking"""
    
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print("Model classes:", model.names)
    
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
    
    # Initialize enhanced tracker
    tracker = EnhancedPlayerTracker(
        max_disappeared=45,   # Longer disappearance tolerance
        max_dormant=200,      # Keep dormant objects longer
        max_distance=100      # Match distance threshold
    )
    
    # Enhanced color scheme
    colors = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue  
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
        (255, 192, 203),  # Pink
        (165, 42, 42),    # Brown
    ]
    
    tracking_data = []
    frame_count = 0
    
    print("Processing video with enhanced persistent tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        results = model(frame, verbose=False)
        
        # Filter detections
        detections = filter_detections(
            results, 
            confidence_threshold=0.4,
            min_area=1000,
            max_area=35000
        )
        
        # Update tracker with frame for appearance extraction
        objects = tracker.update(detections, frame)
        
        # Draw tracking results
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            det_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find best matching tracked object
            best_match_id = None
            min_score = float('inf')
            
            for obj_id, obj_centroid in objects.items():
                distance = math.sqrt(
                    (det_centroid[0] - obj_centroid[0])**2 + 
                    (det_centroid[1] - obj_centroid[1])**2
                )
                
                if distance < min_score and distance < 80:
                    min_score = distance
                    best_match_id = obj_id
            
            # Draw if matched
            if best_match_id is not None:
                color = colors[best_match_id % len(colors)]
                
                # Draw bounding box with thickness based on confidence
                thickness = max(2, int(conf * 4))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Enhanced label with confidence
                label = f"Player {best_match_id} ({conf:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, text_thickness
                )
                
                # Background rectangle for text
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_height - 12), 
                    (x1 + text_width + 8, y1 - 2), 
                    color, 
                    -1
                )
                
                # Text
                cv2.putText(
                    frame, label, (x1 + 4, y1 - 6), 
                    font, font_scale, (255, 255, 255), text_thickness
                )
                
                # Draw center point
                cv2.circle(frame, det_centroid, 5, color, -1)
                cv2.circle(frame, det_centroid, 8, (255, 255, 255), 2)
                
                # Store enhanced tracking data
                tracking_data.append({
                    'frame': frame_count,
                    'player_id': int(best_match_id),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'centroid': det_centroid,
                    'area': (x2-x1) * (y2-y1),
                    'match_distance': float(min_score)
                })
        
        # Draw debug information
        all_objects = tracker.get_all_objects()
        debug_text = f"Frame: {frame_count} | Active: {len(all_objects['active'])} | Dormant: {len(all_objects['dormant'])}"
        cv2.putText(frame, debug_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            active_count = len(all_objects['active'])
            dormant_count = len(all_objects['dormant'])
            print(f"Progress: {progress:.1f}% | Active: {active_count} | Dormant: {dormant_count}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save detailed results
    results_path = output_path.replace('.mp4', '_enhanced_results.json')
    
    # Convert all data to JSON-serializable format
    def convert_to_json_serializable(obj):
        """Convert numpy types and other non-serializable types to Python native types"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert tracking data
    serializable_tracking_data = []
    for item in tracking_data:
        serializable_item = convert_to_json_serializable(item)
        serializable_tracking_data.append(serializable_item)
    
    # Add tracker profile data to results
    final_results = {
        'tracking_data': serializable_tracking_data,
        'object_profiles': {}
    }
    
    # Convert profiles to JSON-serializable format
    for obj_id, profile in tracker.object_profiles.items():
        json_profile = {
            'creation_frame': int(profile['creation_frame']),
            'last_seen_frame': int(profile['last_seen_frame']),
            'total_detections': int(profile['total_detections']),
            'avg_size': (float(profile['avg_size'][0]), float(profile['avg_size'][1])),
            'avg_area': float(profile['avg_area']),
            'position_variance': float(profile['position_variance'])
        }
        final_results['object_profiles'][str(obj_id)] = json_profile
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print comprehensive summary
    unique_players = set(item['player_id'] for item in tracking_data)
    print(f"\n✅ Enhanced Processing Complete!")
    print(f"📹 Output video: {output_path}")
    print(f"📊 Results file: {results_path}")
    print(f"🎯 Frames processed: {frame_count}")
    print(f"👥 Unique players tracked: {len(unique_players)} - IDs: {sorted(unique_players)}")
    print(f"📍 Total detections: {len(tracking_data)}")
    
    # Player persistence analysis
    if tracking_data:
        frame_counts = {}
        for item in tracking_data:
            pid = item['player_id']
            frame_counts[pid] = frame_counts.get(pid, 0) + 1
        
        print(f"\n📈 Player Persistence Analysis:")
        for pid in sorted(frame_counts.keys()):
            persistence = (frame_counts[pid] / frame_count) * 100
            print(f"   Player {pid}: {frame_counts[pid]} frames ({persistence:.1f}% of video)")
    
    return tracking_data


def main():
    parser = argparse.ArgumentParser(description='Enhanced Player Re-Identification with Persistence')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--model', required=True, help='Path to YOLOv11 model')
    parser.add_argument('--output', default='output_enhanced.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Enhanced Player Re-Identification with Persistence...")
        start_time = time.time()
        
        tracking_data = process_video_enhanced(args.video, args.model, args.output)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n⏱️  Processing completed in {processing_time:.2f} seconds")
        print("✅ Success! Your players should now maintain consistent IDs even when leaving and returning to frame.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())