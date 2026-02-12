# Football (Analysis & Tracking) with Top-Down Map

## Description

This project analyzes football match videos to detect players, referees, and the ball using **YOLOv8**.
It classifies players into teams by shirt color, identifies referees, tracks the ball using a **Kalman Filter**, determines ball possession based on proximity, and draws a **Top-Down Map** showing player and ball positions.

The output includes a stabilized, annotated video and a top-down tactical map for visualization.

---

## Features

* **Player & Ball Detection:** Detects players and the ball using YOLOv8.
* **Ball Tracking:** Uses a **Kalman Filter** to predict ball positions when detection is temporarily missing.
* **Team Classification:** Distinguishes teams (Blue / White) and referees (Ref) using HSV analysis of shirt color.
* **Ball Possession Tracking:** Assigns ball possession to the nearest player with smoothing to avoid flickering.
* **Top-Down Tactical Map:** Perspective transformation of the field showing player and ball positions.
* **Visual Output:** Annotated video with bounding boxes for players/referees, circles for the ball, and player IDs.

---

## Technical Details

| Feature                 | Method                                                                | Notes / Accuracy                                 |
| ----------------------- | --------------------------------------------------------------------- | ------------------------------------------------ |
| Player & Ball Detection | `YOLOv8(model)(frame)`                                                | Confidence threshold 0.2                         |
| Ball Tracking           | `BallTrackerKF` class using Kalman Filter                             | Predicts ball position when temporarily lost     |
| Team Classification     | HSV-based shirt analysis + frame memory                               | Majority vote over 25 frames to stabilize        |
| Referee Identification  | HSV check + shirt color region                                        | Detects referees as "REF"                        |
| Ball Possession         | Nearest player to ball using Euclidean distance                       | Smoothed via previous possessor frames           |
| Top-Down Map            | Perspective transformation using OpenCV `cv2.getPerspectiveTransform` | Field mapped to 700x500 pixels                   |
| Output Video            | OpenCV rectangles & circles                                           | Players: Blue/Red, Ref: Yellow, Possessor: White |

---

## Requirements

* **Python:** 3.10+
* **Libraries:**

```bash
pip install opencv-python numpy ultralytics
```

> Note: No external tracker like Norfair is needed; player tracking uses YOLOv8 built-in tracker (`bytetrack.yaml`) and the ball uses Kalman Filter.

---

## Configuration Parameters

| Parameter            | Description                                            | Default / Example              |
| -------------------- | ------------------------------------------------------ | ------------------------------ |
| `MODEL_PATH`         | YOLOv8 model path                                      | `"yolov8n.pt"`                 |
| `VIDEO_PATH`         | Input video path                                       | `"E:\\Sport\\video.mp4"`       |
| `OUTPUT_VIDEO`       | Annotated output video path                            | `"football_SportAI_Final.avi"` |
| `OUTPUT_MAP_VIDEO`   | Top-Down Map output video path                         | `"football_TopDown_Final.avi"` |
| `PLAYER_CLASS`       | YOLO class ID for players                              | 0                              |
| `BALL_CLASS`         | YOLO class ID for the ball                             | 32                             |
| `POSSESSION_DIST`    | Max distance (pixels) to assign possession             | 70                             |
| `MAX_MISSING_FRAMES` | Number of frames the ball can be missing before hiding | 10                             |
| `MAP_W`              | Width of Top-Down Map                                  | 700                            |
| `MAP_H`              | Height of Top-Down Map                                 | 500                            |

---

## Code Workflow

### 1. Imports & Configuration

* Load OpenCV, NumPy, YOLOv8, and helper classes.
* Define video paths, YOLO class IDs, thresholds, and map dimensions.

### 2. Helper Functions

* `center(box)` → computes center of a bounding box.
* `is_white_jersey(crop)` → detects white team shirt.
* `is_sky_blue(crop)` → detects referee shirt.
* `stable_role(track_id, new_role)` → stabilizes team/referee classification using a memory of previous frames.

### 3. Ball Tracking

* `BallTrackerKF` class implements a **Kalman Filter**:

  * `update(pos)` → updates with measured position or predicts if ball not detected.
  * Keeps track of visibility (`visible`) and missing frames.

### 4. Perspective Transformation

* Maps the original video coordinates to a **Top-Down Map** using `cv2.getPerspectiveTransform`.
* Map size: 700x500 pixels.

### 5. Main Loop

1. Read each frame from video.
2. Detect objects (players & ball) using YOLOv8.
3. Separate detections: players, referees, and ball.
4. Classify player roles (team/referee) using HSV analysis and frame memory.
5. Track ball using Kalman Filter to smooth missing detections.
6. Determine ball possession by nearest player, smooth over frames.
7. Draw bounding boxes and IDs for players, referees, possessor, and ball.
8. Draw Top-Down Map with player & ball positions.
9. Write both annotated frame and map frame to output videos.
10. Display frames (optional).

### 6. Cleanup

* Release video capture and writers.
* Close OpenCV windows.

---

## Output

* **Annotated Video:** Shows players, referees, ball, IDs, and possession highlights.
* **Top-Down Map Video:** Shows player and ball positions on a tactical map with color-coded teams.

