# Football Analysis & Tracking

## Description

This project analyzes football match videos to detect players, referees, and the ball using **YOLOv8** and **Norfair Tracker**.
It classifies players into teams by shirt color, identifies referees, tracks the ball using a **Kalman Filter**, and determines ball possession based on proximity.

The output is a stabilized, annotated video showing players, referees, ball, and unique player IDs for clear visualization.

---

## Features

* **Player & Ball Detection:** Detects players and the ball using YOLOv8.
* **Tracking:** Maintains unique IDs across frames using **Norfair Tracker**.
* **Team Classification:** Distinguishes teams by shirt color (Blue, White) and identifies referees (Ref) using HSV analysis.
* **Ball Prediction:** Stabilizes ball position with a **Kalman Filter** when detections are missing.
* **Ball Possession Tracking:** Determines which player is in possession using Euclidean distance and smoothing over frames.
* **Visual Output:** Annotated video with rectangles for players/referees, circles for the ball, and player IDs.

---

## Technical Details

| Feature                 | Method                                                    | Notes / Accuracy                                   |
| ----------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| Player & Ball Detection | `YOLOv8(model)(frame)`                                    | Conf threshold 0.4                                 |
| Tracking                | `Norfair Tracker(distance_threshold=35)`                  | Smooths IDs across frames                          |
| Team Classification     | `classify_team(frame, bbox, track_id)` using HSV          | Majority voting over 10 frames                     |
| Referee Identification  | HSV check + shirt color + region                          | Detects referees as "REF"                          |
| Ball Tracking           | `BallKalman` class using Kalman Filter                    | Predicts ball when temporarily lost                |
| Ball Possession         | Nearest player to ball over `POSSESSION_CONFIRM_FRAMES=8` | Reduces flickering, smooth possession              |
| Output Video            | OpenCV rectangles & circles                               | Players: Blue/White, Ref: Yellow, Possessor: White |

---

## Requirements

* **Python:** 3.10+
* **Libraries:**

```bash
pip install opencv-python numpy ultralytics norfair
```

---

## Configuration Parameters

| Parameter                   | Description                            | Default / Example                 |
| --------------------------- | -------------------------------------- | --------------------------------- |
| `MODEL_PATH`                | YOLOv8 model path                      | `"yolov8n.pt"`                    |
| `VIDEO_PATH`                | Input video path                       | `"E:\\Sport\\tactical video.mp4"` |
| `OUTPUT_VIDEO`              | Output annotated video path            | `"football_SportAI.avi"`          |
| `PLAYER_CLASS`              | YOLO class ID for players              | 0                                 |
| `BALL_CLASS`                | YOLO class ID for the ball             | 32                                |
| `CONF_THRESHOLD`            | Minimum confidence for detections      | 0.4                               |
| `DISTANCE_THRESHOLD`        | Tracker distance threshold             | 35                                |
| `POSSESSION_THRESHOLD`      | Max distance to assign ball possession | 60                                |
| `TEAM_LOCK_FRAMES`          | Number of frames to confirm team color | 10                                |
| `POSSESSION_CONFIRM_FRAMES` | Number of frames to confirm possession | 8                                 |

---

## Code Workflow

### 1. Imports & Config

* Load OpenCV, NumPy, YOLOv8, Norfair, and helper classes.
* Define paths, class IDs, thresholds, and tracker parameters.

### 2. Ball Kalman Filter

* Stabilizes ball coordinates between frames.
* Methods: `update(x, y)` → update & predict, `predict_only()` → predict without update.

### 3. Team Classification

* HSV-based shirt color detection:

  * **Referee:** 85 < H < 115, S > 40 → `"REF"`
  * **White Shirt:** V > 170 & S < 60 → `"WHITE"`
  * Otherwise → `"BLUE"`
* Uses a frame-memory and majority vote over `TEAM_LOCK_FRAMES` to avoid flickering.

### 4. Main Loop

1. Read video frame by frame.
2. Detect objects with YOLOv8.
3. Separate detections for players and ball.
4. Track players & ball using **Norfair Tracker**.
5. Update ball position using **Kalman Filter** if missing.
6. Assign team labels to players.
7. Determine ball possession using nearest player and frame smoothing.
8. Draw bounding boxes, IDs, and ball circle.
9. Save frame to output video and optionally display live.

### 5. Drawing & Output

* **Players:** Blue or White rectangles
* **Referees:** Yellow rectangle with label `"REF"`
* **Player in Possession:** White rectangle (overrides team color)
* **Ball:** Green circle
* **Player IDs:** Shown above bounding boxes

### 6. Cleanup

* Release video and output writer.
* Close OpenCV windows.

