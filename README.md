# Football Player & Ball Tracking 

## Description

This project analyzes football match videos to detect players, referees, and the ball using **YOLOv8**.
It classifies players into teams by shirt color, identifies referees based on position, and tracks which player is in possession of the ball.
The output is an annotated video showing all players, referees, the ball, and player IDs for clear visualization.

---

## Features

* **Player & Ball Detection:** Detects all players and the ball in each video frame using YOLOv8.
* **Player ID Tracking:** Maintains a unique ID for each player across frames to avoid reassignment.
* **Team Classification:** Distinguishes Team A vs Team B using shirt color analysis in HSV color space.
* **Referee Identification:** Detects referees based on position (outside pitch or near sidelines) and shirt color.
* **Ball Possession Tracking:** Determines the player closest to the ball using Euclidean distance.
* **Visual Output:** Annotated video showing players, referees, the ball, and their IDs.

---

## Technical Details

| Feature                 | Method                                                    | Accuracy / Notes                               |
| ----------------------- | --------------------------------------------------------- | ---------------------------------------------- |
| Player & Ball Detection | `model.track(frame, persist=True, conf=0.35, iou=0.6)`    | 85-95% depending on video quality              |
| ID Tracking             | `results.boxes.id`                                        | Ensures continuity across frames               |
| Team Classification     | `classify_team(shirt)` using HSV & white ratio            | ~90% accuracy if shirts are distinct           |
| Referee Identification  | `inside_pitch()` & `near_sideline()`                      | Accurate if referees are outside crowded areas |
| Ball Possession         | `euclidean(ball_center, player_center) < POSSESSION_DIST` | High accuracy within set distance threshold    |
| Output Video            | `cv2.rectangle`, `cv2.putText`                            | Clear visual annotations for all entities      |

---

## Requirements

* **Python:** 3.10+
* **Libraries:**

  * OpenCV (`opencv-python`)
  * Numpy (`numpy`)
  * Ultralytics YOLOv8 (`ultralytics`)
  * SciPy (`scipy`)

```bash
pip install opencv-python numpy ultralytics scipy
```

---

## Configuration Parameters

| Parameter         | Description                              | Default / Example |
| ----------------- | ---------------------------------------- | ----------------- |
| `BALL_CLASS`      | YOLO class ID for the ball               | 32                |
| `PLAYER_CLASS`    | YOLO class ID for players                | 0                 |
| `POSSESSION_DIST` | Max distance to consider ball possession | 60 pixels         |
| `WHITE_THRESHOLD` | Ratio of white to classify Team A        | 0.15              |

> Adjust these values according to video quality, pitch size, and team jersey colors.

---

## Code Structure

### 1. Imports & Config

* Load all required libraries and define constants for model path, video path, classes, and thresholds.

### 2. Model & Video Initialization

* Load YOLOv8 model.
* Open the input video and initialize the output video writer.

### 3. Helper Functions

* `center(box)` → Returns the center of a bounding box.
* `classify_team(crop)` → Determines team based on shirt color.
* `inside_pitch(cx, cy)` → Checks if a player is within pitch bounds.
* `near_sideline(cx, cy)` → Checks if a player is near pitch boundaries (for referee detection).

### 4. Main Loop

* Read video frame by frame.
* Detect and track players, referees, and ball.
* Assign IDs to each detected player.

### 5. Detection Boxes & ID Extraction

* Extract bounding boxes, class IDs, and tracking IDs from YOLO outputs.
* Calculate player and ball centers for tracking.

### 6. Possession Logic

* Determine which player is closest to the ball within `POSSESSION_DIST`.
* Mark that player as having possession.

### 7. Drawing & Output

* Draw rectangles around players and referees, circles for the ball.
* Annotate each player with their ID and highlight the possessor.
* Save each processed frame to the output video.

### 8. Cleanup

* Release video resources and close all OpenCV windows.

---

## Visual Output

* **Team A Players:** Red rectangles
* **Team B Players:** Blue rectangles
* **Player in Possession:** Red rectangle regardless of team
* **Referees:** Yellow rectangle with label "REF"
* **Ball:** Green circle
* **Player IDs:** Displayed above each bounding box
