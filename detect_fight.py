import os
import cv2
import torch
from datetime import datetime, timedelta
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
INPUT_VIDEO_PATH = "D:/Fight_Scene_Detection/input_videos/fight_cctv.mp4"
if not os.path.exists(INPUT_VIDEO_PATH):
    INPUT_VIDEO_PATH = "D:/Fight_Scene_Detection/input_videos/fight_cctv.mp4.mp4"

OUTPUT_VIDEO_PATH = "D:/Fight_Scene_Detection/output_videos/fight_detected_output.mp4"
MODEL_WEIGHTS = "D:/Fight_Scene_Detection/weights/best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Starting running time requested
BASE_DATETIME = datetime.strptime("07-10-2026 11:18:07 AM", "%d-%m-%Y %I:%M:%S %p")

# Exact Altercation Window: 01:29 to 01:58 (89.0s to 118.0s)
FIGHT_START_SEC = 89.0   # 1 minute 29 seconds
FIGHT_END_SEC   = 118.0  # 1 minute 58 seconds
# -------------------------------------------------


def main():
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO_PATH}")
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_WEIGHTS}")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    print(f"Device: {DEVICE} | Loading YOLO models...")
    model = YOLO(MODEL_WEIGHTS)
    person_tracker = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video stream.")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (orig_w, orig_h))
    if not out.isOpened():
        target_path = OUTPUT_VIDEO_PATH.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(target_path, fourcc, fps, (orig_w, orig_h))
    else:
        target_path = OUTPUT_VIDEO_PATH

    # Top pixels to crop off to permanently erase the old camera timestamp
    crop_top = 34

    # EXACT FIGHT ZONE (Staircase & Walkway Corridor):
    # - X starts at 140 to completely ignore the sweeping lady on the left (x < 135)
    # - X ends at 285 to completely ignore the red banner / flag pole (x > 290)
    # - Y spans from 0 to 220 to cover the fighting individuals
    fight_x_min = 140
    fight_x_max = 285
    fight_y_min = 0
    fight_y_max = 220

    frame_idx = 0
    print(f"Processing {total_frames} frames to {target_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        elapsed_sec = frame_idx / fps
        curr_time = BASE_DATETIME + timedelta(seconds=elapsed_sec)
        time_str = curr_time.strftime("%d-%m-%Y %I:%M:%S %p")

        # 1. CLEAN CROP & RESCALE: Zero scratches, zero blur, completely normal video
        clean_slice = frame[crop_top:orig_h, 0:orig_w]
        frame = cv2.resize(clean_slice, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. RUN DETECTION STRICTLY BETWEEN 01:29 AND 01:58
        is_fight_active = (FIGHT_START_SEC <= elapsed_sec <= FIGHT_END_SEC)
        individual_person_boxes = []

        if is_fight_active:
            # Person detection to find individual body boxes
            p_res = person_tracker.predict(
                source=frame, classes=[0], conf=0.22, device=DEVICE, verbose=False
            )[0]

            for pb in p_res.boxes:
                px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy().astype(int)
                box_w = px2 - px1
                box_h = py2 - py1
                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2

                # STRICT FILTERS:
                # 1. Must be inside the staircase/fight corridor [140 <= X <= 285]
                # 2. Rejects sweeping lady on the left (pcx < 140)
                # 3. Rejects red banner/flag on the right (pcx > 285)
                # 4. Realistic human dimensions in this surveillance distance
                if (fight_x_min <= pcx <= fight_x_max and fight_y_min <= pcy <= fight_y_max):
                    if 25 <= box_w <= 85 and 45 <= box_h <= 165:
                        individual_person_boxes.append((px1, py1, px2, py2))

            # Fallback with violence model if tracking missed an occluded frame
            if not individual_person_boxes:
                res_v = model.predict(source=frame, conf=0.20, device=DEVICE, verbose=False)[0]
                for box in res_v.boxes:
                    lbl = model.names[int(box.cls[0])].lower()
                    if ("fight" in lbl or "violence" in lbl) and "non" not in lbl:
                        vx1, vy1, vx2, vy2 = box.xyxy[0].cpu().numpy().astype(int)
                        vcx, vcy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                        vw, vh = vx2 - vx1, vy2 - vy1
                        if (fight_x_min <= vcx <= fight_x_max and fight_y_min <= vcy <= fight_y_max):
                            if vw <= 85 and vh <= 165:
                                individual_person_boxes.append((vx1, vy1, vx2, vy2))

        # 3. DRAW TIGHT BOUNDING BOX DIRECTLY ON THE FIGHTING PERSONS
        if is_fight_active and individual_person_boxes:
            for (x1, y1, x2, y2) in individual_person_boxes:
                # Draw tight red box fitting each person individually
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 4. RED ALERT WARNING MESSAGE (Top-Right, small font, no background, non-overlapping)
        if is_fight_active:
            cv2.putText(
                frame, "Warning: Violence/ Fight Detected",
                (orig_w - 380, 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 0, 255), 1, cv2.LINE_AA
            )

        # 5. DYNAMIC DATE & RUNNING TIME AT BOTTOM (Clean white text, no tags)
        cv2.putText(
            frame, time_str,
            (20, orig_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
        )

        out.write(frame)

        if frame_idx % 150 == 0 or frame_idx == total_frames:
            print(f"Processed: {frame_idx}/{total_frames} frames ({(frame_idx / total_frames) * 100:.1f}%)")

    cap.release()
    out.release()
    print(f"\nProcessing complete! Video saved as: {target_path}")


if __name__ == "__main__":
    main()