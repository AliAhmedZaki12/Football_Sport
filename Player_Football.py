# ============================================================
# Football Analysis - Importing (Libraries & Configs)
# ============================================================

import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Tracker, Detection
from collections import defaultdict, Counter


MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "E:\\Sport\\tactical video.mp4"
OUTPUT_VIDEO = "football_SportAI.avi"

PLAYER_CLASS = 0
BALL_CLASS = 32

CONF_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 35
POSSESSION_THRESHOLD = 60
TEAM_LOCK_FRAMES = 10
POSSESSION_CONFIRM_FRAMES = 8

# ============================================================
# Ball Kalman Filter
# ============================================================

class BallKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])

    def predict_only(self):
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])

# ============================================================
# Team Classification 
# ============================================================

team_memory = defaultdict(list)
team_locked = {}

def classify_team(frame, bbox, track_id):
    if track_id in team_locked:
        return team_locked[track_id]

    x1, y1, x2, y2 = map(int, bbox)
    y1s = int(y1 + (y2 - y1) * 0.2)
    y2s = int(y1 + (y2 - y1) * 0.45)
    crop = frame[y1s:y2s, x1:x2]

    if crop.size == 0:
        return "BLUE"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    median_hsv = np.median(hsv.reshape(-1, 3), axis=0)
    h, s, v = median_hsv

    if 85 < h < 115 and s > 40:
        label = "REF"
    elif v > 170 and s < 60:
        label = "WHITE"
    else:
        label = "BLUE"

    team_memory[track_id].append(label)
    if len(team_memory[track_id]) >= TEAM_LOCK_FRAMES:
        majority = Counter(team_memory[track_id]).most_common(1)[0][0]
        team_locked[track_id] = majority
        return majority

    return label

# ============================================================
# MAIN
# ============================================================

def main():
    model = YOLO(MODEL_PATH)

    tracker_players = Tracker(distance_function="euclidean", distance_threshold=DISTANCE_THRESHOLD)
    tracker_ball = Tracker(distance_function="euclidean", distance_threshold=30)
    kalman = BallKalman()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    possession_candidate = None
    possession_counter = 0
    confirmed_possession = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        detections_players = []
        detections_ball = []

        # ---------------------------
        # Detection
        # ---------------------------
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if conf < CONF_THRESHOLD:
                continue

            if cls == PLAYER_CLASS:
                detections_players.append(Detection(points=np.array([[x1,y1],[x2,y2]])))
            elif cls == BALL_CLASS:
                detections_ball.append(Detection(points=np.array([[x1,y1],[x2,y2]])))

        # ---------------------------
        # Tracking
        # ---------------------------
        tracked_players = tracker_players.update(detections_players)
        tracked_balls = tracker_ball.update(detections_ball)

        players = []
        ball_center = None

        for obj in tracked_players:
            x1, y1 = obj.estimate[0]
            x2, y2 = obj.estimate[1]
            team = classify_team(frame, (x1, y1, x2, y2), obj.id)
            players.append((obj.id, (x1, y1, x2, y2), team))

        if tracked_balls:
            obj = tracked_balls[0]
            x1, y1 = obj.estimate[0]
            x2, y2 = obj.estimate[1]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            ball_center = kalman.update(cx, cy)
        else:
            if kalman.initialized:
                ball_center = kalman.predict_only()

        # ---------------------------
        # Possession Smoothing
        # ---------------------------
        if ball_center:
            min_dist = POSSESSION_THRESHOLD
            current_candidate = None

            for pid, bbox, team in players:
                if team == "REF":
                    continue
                px = int((bbox[0] + bbox[2]) / 2)
                py = int((bbox[1] + bbox[3]) / 2)
                dist = np.linalg.norm(np.array([px, py]) - np.array(ball_center))
                if dist < min_dist:
                    min_dist = dist
                    current_candidate = pid

            if current_candidate == possession_candidate:
                possession_counter += 1
            else:
                possession_candidate = current_candidate
                possession_counter = 0

            if possession_counter >= POSSESSION_CONFIRM_FRAMES:
                confirmed_possession = possession_candidate

        # ---------------------------
        # Drawing
        # ---------------------------
        for pid, bbox, team in players:
            x1, y1, x2, y2 = map(int, bbox)
            if team == "REF":
                color = (0, 255, 255)
            elif team == "WHITE":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            if pid == confirmed_possession:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if ball_center:
            cx, cy = ball_center
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        out.write(frame)
        cv2.imshow("Stable Football Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
