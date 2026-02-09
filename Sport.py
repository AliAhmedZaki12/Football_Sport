# ===============================
# Importing (Libraries & Configs)
# ===============================
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import euclidean

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = r"E:\Sport\video_2026-02-07_23-44-07.mp4"
OUTPUT_VIDEO = "football_Sport.avi"

BALL_CLASS = 32
PLAYER_CLASS = 0

POSSESSION_DIST = 60
WHITE_THRESHOLD = 0.15

# ===============================
# (Model & Video Initialization)
# ===============================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (w, h)
)

# ===============================
# (Helper Functions)
# ===============================
def center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def classify_team(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 60, 255])
    )
    ratio = np.sum(mask > 0) / mask.size
    return "A" if ratio > WHITE_THRESHOLD else "B"

def inside_pitch(cx, cy):
    return (
        w * 0.05 < cx < w * 0.95 and
        h * 0.08 < cy < h * 0.95
    )

def near_sideline(cx, cy):
    return (
        cx < w * 0.07 or cx > w * 0.93 or
        cy < h * 0.10 or cy > h * 0.92
    )

# ===============================
# MAIN LOOP
# ===============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    players = []
    ball_center = None

    results = model.track(
        frame,
        persist=True,
        conf=0.35,
        iou=0.6
    )[0]
    # ===============================
    #(Detection Boxes & ID Extraction)
    # ===============================
    if results.boxes.id is not None:
        for box, cls, tid in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.id.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = center((x1, y1, x2, y2))

            # -------- BALL --------
            if int(cls) == BALL_CLASS:
                ball_center = (cx, cy)
                cv2.circle(frame, ball_center, 5, (0, 255, 0), -1)
                continue

            # -------- PLAYER / REF --------
            if int(cls) != PLAYER_CLASS:
                continue

            # crop shirt area
            y2s = int(y1 + (y2 - y1) * 0.4)
            shirt = frame[y1:y2s, x1:x2]
            if shirt.size == 0:
                continue

            team = classify_team(shirt)

            # referee logic
            if (not inside_pitch(cx, cy)) or (near_sideline(cx, cy) and team == "B"):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "REF",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )
                continue

            # real player
            players.append({
                "id": int(tid),
                "team": team,
                "center": (cx, cy),
                "box": (x1, y1, x2, y2)
            })

    # ===============================
    # POSSESSION
    # ===============================
    possessor = None
    if ball_center:
        min_d = POSSESSION_DIST
        for p in players:
            d = euclidean(ball_center, p["center"])
            if d < min_d:
                min_d = d
                possessor = p

    # ===============================
    # DRAW PLAYERS
    # ===============================
    for p in players:
        x1, y1, x2, y2 = p["box"]

        # Team A = Red / Team B = Blue
        color = (0, 0, 255) if p["team"] == "A" else (255, 0, 0)

        if possessor and p["id"] == possessor["id"]:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {p['id']}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        # Final Processing 

    out.write(frame)
    cv2.imshow("Football Analysis", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
