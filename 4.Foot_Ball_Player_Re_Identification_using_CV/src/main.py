import cv2
import argparse
from player_detector import PlayerDetector
from player_tracker import PlayerTracker
import json

"""
python src/main.py --video data/videos/15sec_input_720p.mp4 --model data/models/best.pt
"""

def process_video(video_path, model_path, output_path):
    """Process video for player re-identification"""
    
    # Initialize detector and tracker
    detector = PlayerDetector(model_path)
    tracker = PlayerTracker(max_disappeared=30, max_distance=80)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracking_results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect players
        detections = detector.detect_players(frame)
        
        # Update tracker
        objects = tracker.update(detections)
        
        # Draw bounding boxes and IDs
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf = detection.astype(int)
            
            # Find corresponding tracked object
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            closest_id = None
            min_dist = float('inf')
            
            for obj_id, (obj_cx, obj_cy) in objects.items():
                dist = ((cx - obj_cx)**2 + (cy - obj_cy)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_id = obj_id
            
            if closest_id is not None and min_dist < 50:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Player {closest_id}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Store results
                tracking_results.append({
                    'frame': frame_count,
                    'player_id': int(closest_id),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    # Save tracking results
    with open('outputs/tracking_results.json', 'w') as f:
        json.dump(tracking_results, f, indent=2)
    
    print(f"Processing complete! Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--model', required=True, help='YOLO model path')
    parser.add_argument('--output', default='outputs/output_video.mp4', help='Output video path')
    
    args = parser.parse_args()
    process_video(args.video, args.model, args.output)