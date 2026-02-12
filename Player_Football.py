# ===============================
# Football Player (Tracking  & Analytics)
# ===============================

# ===============================
# IMPORTS
# ===============================
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from collections import defaultdict, deque

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "yolov8n.pt"                             # YOLOv8 model path
VIDEO_PATH = r"E:\Sport\video_2026-02-07_23-44-07.mp4"
OUTPUT_VIDEO = "football_SportAI_Final.avi"          # Annotated output video
OUTPUT_MAP_VIDEO = "football_TopDown_Final.avi"      # Top-Down map video

BALL_CLASS = 32                                      # YOLO class ID for ball
PLAYER_CLASS = 0                                     # YOLO class ID for players
POSSESSION_DIST = 70                                 # Distance threshold for possession detection
MAX_MISSING_FRAMES = 10                              # Frames allowed without detecting ball

# Memory for stabilizing team classification per player
track_memory = defaultdict(lambda: deque(maxlen=25))

# ===============================
# LOAD YOLO MODEL
# ===============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

# Video writer for annotated video
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)

# Video writer for Top-Down map
MAP_W, MAP_H = 700, 500
map_out = cv2.VideoWriter(
    OUTPUT_MAP_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (MAP_W, MAP_H)
)

# ===============================
# HELPER FUNCTIONS
# ===============================
def center(box):
    """
    Returns the center coordinates (x,y) of a bounding box.
    """
    x1, y1, x2, y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def is_white_jersey(crop):
    """
    Detects if the shirt area corresponds to a white jersey using HSV threshold.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.logical_and(s < 75, v > 155)
    ratio = np.sum(mask) / mask.size
    return ratio > 0.05

def is_sky_blue(crop):
    """
    Detects if the shirt area corresponds to sky-blue referee jersey using HSV threshold.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85,40,120]), np.array([120,255,255]))
    ratio = np.sum(mask>0) / mask.size
    return ratio > 0.06

def stable_role(track_id, new_role):
    """
    Stabilizes the role classification of a player using memory of previous roles.
    """
    memory = track_memory[track_id]
    memory.append(new_role)
    if memory.count("A") > 6: return "A"
    if memory.count("REF") > 4: return "REF"
    return max(set(memory), key=memory.count)

# ===============================
# BALL TRACKER USING KALMAN FILTER
# ===============================
class BallTrackerKF:
    """
    Kalman Filter based tracker for the football to smooth ball movement.
    - Predicts ball position when not detected.
    - Keeps track of visibility.
    """
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
        self.predicted = None
        self.missing_frames = 0
        self.visible = False

    def update(self, pos=None):
        """
        Updates the Kalman filter with a new position (if available).
        Returns the predicted or corrected position of the ball.
        """
        if pos is not None:
            measurement = np.array([[np.float32(pos[0])],[np.float32(pos[1])]])
            self.kalman.correct(measurement)
            self.missing_frames = 0
            self.visible = True
        else:
            self.missing_frames += 1
            if self.missing_frames > MAX_MISSING_FRAMES:
                self.visible = False

        self.predicted = self.kalman.predict()
        x, y = self.predicted[:2].flatten()
        return int(x), int(y)

# Initialize ball tracker and possession smoothing
ball_tracker = BallTrackerKF()
possessor_smooth = None

# ===============================
# PERSPECTIVE TRANSFORMATION (TOP-DOWN MAP)
# ===============================
src_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])         # Video frame corners
dst_pts = np.float32([[0,0],[MAP_W,0],[0,MAP_H],[MAP_W,MAP_H]])  # Map corners
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# ===============================
# MAIN LOOP: VIDEO PROCESSING
# ===============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    players = []
    ball_center = None
    infer_frame = frame.copy()

    # -------- YOLO TRACKING --------
    results = model.track(
        infer_frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.20,
        imgsz=1280,
        iou=0.7,
        agnostic_nms=True
    )[0]

    if results.boxes.id is not None:
        for box, cls, tid in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.id.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = center((x1,y1,x2,y2))

            # -------- BALL DETECTION --------
            if int(cls) == BALL_CLASS:
                ball_center = (cx, cy)
                continue

            # Ignore non-player objects
            if int(cls) != PLAYER_CLASS:
                continue

            # Shirt crop for team/referee classification
            y1s = int(y1 + (y2-y1)*0.18)
            y2s = int(y1 + (y2-y1)*0.40)
            shirt = frame[y1s:y2s, x1:x2]
            if shirt.size == 0: continue

            # Detect role
            if is_sky_blue(shirt):
                role = "REF"
            elif is_white_jersey(shirt):
                role = "A"
            else:
                role = "B"

            role = stable_role(int(tid), role)

            # Draw referees
            if role == "REF":
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                cv2.putText(frame,"REF",(x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                continue

            players.append({
                "id": int(tid),
                "team": role,
                "center": (cx,cy),
                "box": (x1,y1,x2,y2)
            })

    # -------- BALL TRACKING --------
    tracked_ball = ball_tracker.update(ball_center)
    if ball_tracker.visible:
        cv2.circle(frame, tracked_ball, 8, (0,255,0), -1)

    # -------- POSSESSION DETECTION --------
    possessor = None
    if tracked_ball:
        min_d = POSSESSION_DIST
        for p in players:
            d = euclidean(tracked_ball, p["center"])
            if d < min_d:
                min_d = d
                possessor = p

    if possessor:
        possessor_smooth = possessor
    else:
        possessor = possessor_smooth  # Keep last possessor if ball temporarily lost

    # -------- DRAW PLAYERS --------
    for p in players:
        x1,y1,x2,y2 = p["box"]
        color = (0,0,255) if p["team"]=="A" else (255,0,0)
        thickness = 2
        if possessor and p["id"] == possessor["id"]:
            color = (255,255,255)  # Highlight possessor
            thickness = 3
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,thickness)
        cv2.putText(frame,f"ID {p['id']}",(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    # -------- DRAW TOP-DOWN MAP --------
    map_frame = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    if ball_tracker.visible:
        bx, by = cv2.perspectiveTransform(
            np.array([[[tracked_ball[0], tracked_ball[1]]]], dtype=np.float32), M
        )[0][0]
        cv2.circle(map_frame, (int(bx), int(by)), 5, (0,255,0), -1)

    for p in players:
        px, py = cv2.perspectiveTransform(
            np.array([[[p["center"][0], p["center"][1]]]], dtype=np.float32), M
        )[0][0]
        col = (0,0,255) if p["team"]=="A" else (255,0,0)
        if possessor and p["id"] == possessor["id"]:
            col = (255,255,255)
        cv2.circle(map_frame, (int(px), int(py)), 5, col, -1)

    # -------- SHOW FRAMES & WRITE VIDEOS --------
    cv2.imshow("Football Analysis", frame)
    cv2.imshow("Top-Down Map", map_frame)

    out.write(frame)
    map_out.write(map_frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

# ===============================
# RELEASE RESOURCES
# ===============================
cap.release()
out.release()
map_out.release()
cv2.destroyAllWindows()

