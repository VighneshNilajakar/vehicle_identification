import cv2
import easyocr
import pyperclip
from plyer import notification
import time
import numpy as np

# EasyOCR initialization
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have a CUDA GPU

def detect_and_annotate(frame):
    """
    Runs OCR on the bottom half of the frame, returns annotated frame and detected text.
    """
    h, w, _ = frame.shape
    roi = frame[h//2:, :]  # bottom half
    results = reader.readtext(roi)

    plate = None
    for (bbox, text, prob) in results:
        if prob < 0.5 or len(text) < 6:
            continue
        # bbox is np array of 4 points, relative to ROI
        pts = np.array(bbox).astype(int)
        pts[:,1] += h//2  # shift y-coords back to full frame
        # draw bounding box
        cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
        # put text
        x, y = pts[0]
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        plate = text
        break  # only first good result

    return frame, plate

def detect_material_color(frame):
    avg = frame.mean(axis=0).mean(axis=0)
    if avg[2] > 150:
        return "Red"
    if avg[1] > 150:
        return "Green"
    return "Unknown"

def send_to_clipboard_and_notify(text):
    pyperclip.copy(text)
    notification.notify(
        title="Vehicle Info",
        message=text,
        timeout=5
    )

def main():
    url = "http://192.168.151.7:8080/video"  # your real stream URL
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: can't open stream")
        return

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream error, retrying...")
            time.sleep(1)
            continue

        # only process every N seconds
        if time.time() - last_time > 5:
            annotated, plate = detect_and_annotate(frame.copy())
            material = detect_material_color(frame)

            if plate:
                info = f"Plate: {plate}, Material: {material}"
                print("[INFO]", info)
                send_to_clipboard_and_notify(info)
            else:
                print("[INFO] No plate detected.")

            last_time = time.time()
        else:
            annotated = frame

        # show live (or fallback)
        try:
            cv2.imshow("Vehicle Inspector", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # fallback: skip display if GUI support missing
            pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
