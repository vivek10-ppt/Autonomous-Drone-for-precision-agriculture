from picamera2 import Picamera2
import cv2
import numpy as np
import time
import struct
from multiprocessing import shared_memory
from tflite_runtime.interpreter import Interpreter
from collections import deque

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "best_int8.tflite"

CONF_THRESHOLD = 0.450
INPUT_SIZE = 320

SHM_NAME = "ml_detection_shm"
STRUCT_FORMAT = "iii"   # detected, x_offset, y_offset
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

CLASS_ID_OF_INTEREST = 0  # single-class model

OUTPUT_VIDEO = "live_monitor.mp4"
OUTPUT_FPS = 20
# =====================================================

# ---------------- Shared Memory Setup ----------------
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=STRUCT_SIZE)
    print("Shared memory created")
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("Shared memory attached")

struct.pack_into(STRUCT_FORMAT, shm.buf, 0, 0, 0, 0)

# ---------------- Load TFLite Model ----------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, input_h, input_w, _ = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

print(f"Model input: {input_w}x{input_h}, dtype={input_dtype}")

# ---------------- Pi Camera Setup ----------------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 640), "format": "BGR888"},
    controls={"FrameRate": 60}
)
picam2.configure(config)
picam2.start()

print("Pi Camera started")

# ---------------- Video Writer ----------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    OUTPUT_FPS,
    (640, 640)
)

print(f"Recording live output to {OUTPUT_VIDEO}")

# ---------------- FPS Tracking ----------------
fps_window = deque(maxlen=30)
prev_time = time.time()

# ---------------- Helper: Convert Box ----------------
def convert_box(det, w_frame, h_frame):
    x1, y1, x2, y2, conf, cls = det

    m = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if m <= 1.5:
        x1 *= w_frame
        x2 *= w_frame
        y1 *= h_frame
        y2 *= h_frame
    else:
        sx = w_frame / input_w
        sy = h_frame / input_h
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

    x1 = int(max(0, min(w_frame - 1, x1)))
    y1 = int(max(0, min(h_frame - 1, y1)))
    x2 = int(max(0, min(w_frame - 1, x2)))
    y2 = int(max(0, min(w_frame - 1, y2)))

    return x1, y1, x2, y2, float(conf), int(cls)

# ---------------- Main Loop ----------------
detected_counter = 0
miss_counter = 0
last_state = 0
last_rel_x = 0
last_rel_y = 0
try:
    while True:
        frame = picam2.capture_array()
        h_frame, w_frame = frame.shape[:2]

        frame_cx = w_frame // 2
        frame_cy = h_frame // 2

        # -------- FPS calculation --------
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_window.append(fps)
        avg_fps = sum(fps_window) / len(fps_window)

        # -------- Preprocess --------
        resized = cv2.resize(frame, (input_w, input_h))

        input_data = resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # -------- Inference --------
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        dets = np.squeeze(output, axis=0)

        best_det = None
        best_conf = 0.0
        rel_x = rel_y = 0
        detected_flag = 0

        for det in dets:
            if det[4] < CONF_THRESHOLD:
                continue
            if int(det[5]) != CLASS_ID_OF_INTEREST:
                continue
            if det[4] > best_conf:
                best_conf = det[4]
                best_det = det

        # ---- Temporal smoothing state machine ----
        if best_det is not None:
            # Detection found in this frame
            x1, y1, x2, y2, conf, cls = convert_box(best_det, w_frame, h_frame)

            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2

            rel_x = bbox_cx - frame_cx
            rel_y = bbox_cy - frame_cy
            detected_flag = 1

            detected_counter += 1
            miss_counter = 0

            # Require 2 consecutive detections before we flip ON
            if detected_counter >= 2:
                last_state = 1
                last_rel_x = rel_x
                last_rel_y = rel_y
                struct.pack_into(STRUCT_FORMAT, shm.buf, 0, 1, int(last_rel_x), int(last_rel_y))

            # Draw on preview
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (bbox_cx, bbox_cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"conf={conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # No detection this frame
            miss_counter += 1
            detected_counter = 0

            # Only turn OFF after 4 consecutive misses
            if miss_counter >= 4:
                last_state = 0
                struct.pack_into(STRUCT_FORMAT, shm.buf, 0, 0, 0, 0)

        # IMPORTANT:
        # If neither threshold met, DO NOT WRITE shared memory keeps previous valid value


        # -------- Overlays --------
        cv2.circle(frame, (frame_cx, frame_cy), 6, (255, 0, 0), -1)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Detected: {detected_flag}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if detected_flag else (0, 0, 255), 2)

        cv2.putText(frame, f"Offset X: {rel_x}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Offset Y: {rel_y}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -------- Save frame --------
        video_out.write(frame)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    picam2.stop()
    video_out.release()
    shm.close()
    # unlink ONLY if no reader is running
    shm.unlink()
    print("Shutdown complete")
