import cv2
import easyocr
import pyperclip
from plyer import notification
import time
import numpy as np

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def detect_number_plate(frame):
    height, width, _ = frame.shape
    roi = frame[int(height * 0.5):, :]  # Crop bottom half of the frame
    results = reader.readtext(roi)
    for (bbox, text, prob) in results:
        if len(text) >= 6 and prob > 0.5:
            return text
    return None

def detect_material_color(frame):
    avg_color = frame.mean(axis=0).mean(axis=0)
    if avg_color[2] > 150:
        return "Red Material"
    elif avg_color[1] > 150:
        return "Green Material"
    else:
        return "Unknown Material"

def send_to_clipboard_and_notify(text):
    pyperclip.copy(text)
    notification.notify(
        title="Vehicle Info Detected",
        message=text,
        timeout=5
    )

def main():
    # Replace with your actual IP camera stream URL
    url = "http://192.168.229.101:8080/video"  # Example: IP Webcam app
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return

    print("Starting video stream...")

    last_processed_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        # Process every 5 seconds
        if time.time() - last_processed_time > 5:
            plate = detect_number_plate(frame)
            material = detect_material_color(frame)

            if plate:
                info = f"Plate: {plate}, Material: {material}"
                print(f"[INFO] {info}")
                send_to_clipboard_and_notify(info)
            else:
                print("[INFO] No plate detected.")

            last_processed_time = time.time()

    cap.release()

if __name__ == "__main__":
    main()
