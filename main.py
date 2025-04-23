import cv2
import numpy as np
import pyperclip
import time
from plyer import notification
from ultralytics import YOLO
import easyocr

# —————————————————————————————
# 1) MODEL LOADING
#—————————————————————————————
# Use the generic YOLOv8n (auto-downloads on first run).
# Later, swap to "yolov8n-plate.pt" once you have that file locally.
yolo_model = YOLO("yolov8n.pt")         # :contentReference[oaicite:1]{index=1}
ocr_reader = easyocr.Reader(['en'], gpu=False)

# —————————————————————————————
# 2) UTILITIES
#—————————————————————————————
def preprocess_plate(roi):
    """Grayscale + adaptive threshold to boost OCR accuracy."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def detect_and_read(frame):
    """
    1) Run YOLO detection → get boxes  
    2) Crop & preprocess each box → EasyOCR  
    Returns (annotated_frame, detected_text or None)
    """
    annotated = frame.copy()
    results = yolo_model(frame)[0]  # returns a Results object :contentReference[oaicite:2]{index=2}
    text_found = None

    for *box, conf, cls in results.boxes.data.tolist():
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        proc = preprocess_plate(roi)
        ocr_res = ocr_reader.readtext(
            proc,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )  # :contentReference[oaicite:3]{index=3}

        if ocr_res:
            txt = ocr_res[0][1]
            # draw box + text
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(annotated, txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            text_found = txt
            break

    return annotated, text_found

def send_to_clipboard_and_notify(text):
    pyperclip.copy(text)  # :contentReference[oaicite:4]{index=4}
    notification.notify(
        title="Number Plate Detected",
        message=text,
        timeout=4
    )

# —————————————————————————————
# 3) MAIN LOOP
#—————————————————————————————
def main():
    stream_url = "http://10.1.232.197:8080/video"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Unable to open stream")
        return

    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # process every 3 seconds
        if time.time() - last_time > 3:
            annotated, plate = detect_and_read(frame)
            if plate:
                print("[INFO] Plate:", plate)
                send_to_clipboard_and_notify(plate)
            else:
                print("[INFO] No plate found.")
            last_time = time.time()
        else:
            annotated = frame

        cv2.imshow("ANPR Preview", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
