# Player Re-Identification in Sports Footage

## Overview
This project implements player re-identification for sports footage using YOLOv11 object detection and custom tracking algorithms.

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## File Structure
1. main.py - Orchestrates the entire pipeline: reads video, detects and tracks players, and saves output.
2. player_detector.py - Uses a YOLO model to detect players in indivisual frames.
3. player_tracker.py - maintaines player identity over time by tracking centroids with custom logic.

## Features 
1. Player Detection:
    Uses Yolo given model to detect players in each frame.
2. Player Tracking:
    A custom object tracker maintains player dentities frame-by-frame by comparing centroid distances.
3. Video Processing:
    Reads input video, overlays bounding boxes and player IDs AND outputs an anootated video with JSON tracking data.
4. Output Generation:
    Annotated video(outputs/output_video.mp4)
    JSON tracking file (outputs/tracking_results.json) with player IDs, bouding boxes, confidence frame number.

## Performance & usage
* Model Used: YOLO (you can plug in any .pt model compatible with ultralytics)
* Tracking Speed: Real-time processing on standard video resolution (720p tested)
* Accurate ID Re-association: Based on proimity of player centroids (distance threshold = 80 pixesl)
* Supports Video Input via CLI:
* python main.py --video path/to/video.mp4 --model path/to/model.pt

## Limitations 
* Logic:- Used Euclidean distance for centroid matching is failed with overlapping players or occlusions.
* Player Class Assumption: model assign class 0 for players. any differet labelling will cause detection failure unless modified in player_detector.py
* No Motion or Appearance Modeling: Unlike advanced trackers like DeepSORT, this implementation doesn't incorporate appearance embeddings or Kalman filters.
* Hardcoded Output paths: Output files (outputs/output_video.mp4, outputs/tracking_results.json) are fixed unless changed via CLI or code.

## 🔮 Future Improvements
* ✅ Integrate a more sophisticated multi-object tracking system (e.g., DeepSORT or BYTETrack)
* ✅ Add support for tracking lost and reappeared players using re-identification features.
* ✅ Visualize player movement paths and heatmaps for sports analytics.
* ✅ Make output directories configurable and improve CLI usability.
* ✅ Add performance benchmarks (FPS, accuracy metrics).
* ✅ Support multi-class tracking (e.g., players, ball, referees).




# Save the generated README content to a file

readme_content = """# 🏃‍♂️ Player Re-Identification in Sports Video

This project implements **multi-stage player tracking** and re-identification in sports videos using YOLO-based detection and custom tracking logic. It supports increasing levels of tracking sophistication, from simple centroid matching to advanced persistent re-identification with object memory and trajectory analysis.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Baseline tracker using YOLO + centroid tracking |
| `simple_player_reid.py` | Basic tracker with distance-based object persistence |
| `Improved_player_reid.py` | Tracker with better filtering and size-based disambiguation |
| `enhanced_player_reid.py` | Advanced tracker with object memory, re-identification, and trajectory analysis |
| `player_detector.py` | Class wrapper for YOLO-based player detection |
| `player_tracker.py` | Basic ID-based centroid tracking |

---

## 🎯 Features

### ✅ Common Across All Versions

- YOLOv8 player detection
- Tracks players across frames
- Draws bounding boxes and IDs
- Saves annotated video + JSON metadata

---

### 🧩 Version-wise Features

#### 1. `main.py` – **Basic Tracker**
- Centroid distance tracking
- ID assignment based on closest object
- Frame-by-frame JSON logging

#### 2. `simple_player_reid.py` – **Simple Tracker**
- Easier modular tracking class
- Confidence-based filtering
- Colored ID annotations
- Supports bounding box and centroid storage

#### 3. `Improved_player_reid.py` – **Improved Tracker**
- Size-based filtering (removes ball)
- Distance matrix for assignment
- Detection area constraints
- Tracks object size + history

#### 4. `enhanced_player_reid.py` – **Enhanced Tracker**
- Persistent object profiles with:
  - Position variance
  - Trajectory (velocity) tracking
  - Size consistency
  - Appearance patch storage (ROI)
- Re-identifies returning players
- Dormant state for players off-camera
- Confidence-weighted ID stability

---

## ⚙️ Performance

| Version | Tracking Stability | Filtering | Re-ID | Notes |
|--------|--------------------|-----------|-------|-------|
| `main.py` | ⭐⭐ | ❌ | ❌ | Suitable for small demos |
| `simple_player_reid.py` | ⭐⭐⭐ | Basic | ❌ | Better ID tracking |
| `Improved_player_reid.py` | ⭐⭐⭐⭐ | Size + Class | ⛔ | Good balance of precision |
| `enhanced_player_reid.py` | ⭐⭐⭐⭐⭐ | ROI + Profile | ✅ | Handles occlusion, exit/reentry |

---

## 🚧 Limitations

- All models assume YOLO class `0` is a **player**.
- Lighting or motion blur can impact detection confidence.
- **No deep appearance re-ID** (no ResNet/Siamese embedding).
- Uses custom logic instead of Hungarian algorithm for assignment.
- Not optimized for real-time on CPU for large inputs.

---

## 🔮 Future Improvements

- Integrate DeepSORT or BoT-SORT for hybrid tracking
- Train custom re-ID model for appearance features
- Replace centroid distance with cosine similarity on embeddings
- Use Kalman Filter for motion prediction
- GUI dashboard to visualize player stats and heatmaps
- Allow ball detection and separation

---

## 🏁 Usage

```bash
# Basic version
python main.py --video path/to/input.mp4 --model path/to/yolo.pt

# Simple re-ID
python simple_player_reid.py --video path/to/input.mp4 --model path/to/yolo.pt

# Improved version
python Improved_player_reid.py --video path/to/input.mp4 --model path/to/yolo.pt

# Enhanced re-ID with memory
python enhanced_player_reid.py --video path/to/input.mp4 --model path/to/yolo.pt
