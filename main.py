import cv2
import numpy as np
import time
from ultralytics import YOLO
import easyocr
import pyperclip
from plyer import notification

# --------------------------------------------------
# Configuration
# --------------------------------------------------
CAMERA_URL = "http://192.168.29.145:8080/video"  # Replace with your IP Webcam stream URL
PLATE_MODEL_PATH = "yolov8n.pt"  # Or a more specific plate detection model if available
OCR_LANGUAGE = "en"
OCR_GPU = False  # Set to True if you have a compatible GPU and CUDA setup
TIME_BETWEEN_FRAMES = 3  # Process every 3 seconds.  Adjust as needed.
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# --------------------------------------------------
# Model Initialization
# --------------------------------------------------
def load_models():
    """Loads the YOLO model for plate detection and the EasyOCR reader."""
    try:
        yolo_model = YOLO(PLATE_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None

    try:
        ocr_reader = easyocr.Reader([OCR_LANGUAGE], gpu=OCR_GPU)
    except Exception as e:
        print(f"Error loading EasyOCR: {e}")
        return None, None
    return yolo_model, ocr_reader

# --------------------------------------------------
# Image Processing
# --------------------------------------------------
def preprocess_plate(plate_image):
    """
    Preprocesses the plate image to improve OCR accuracy.
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # More robust preprocessing:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blurred, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

def detect_plate_and_text(frame, yolo_model, ocr_reader):
    """
    Detects the number plate and reads the text on it using YOLO and EasyOCR.

    Args:
        frame (numpy.ndarray): The input frame.
        yolo_model: The YOLO model.
        ocr_reader: The EasyOCR reader.

    Returns:
        tuple: (annotated frame, detected plate text, plate coordinates)
               Returns (frame, None, None) on error or no detection.
    """
    annotated_frame = frame.copy()
    try:
        results = yolo_model(frame)[0]  # Get the first result
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return annotated_frame, None, None

    if not results or not results.boxes:
        return annotated_frame, None, None  # No detections

    for *box, _, _ in results.boxes.data.tolist():
        x1, y1, x2, y2 = map(int, box)
        plate_coords = (x1, y1, x2, y2)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), GREEN_COLOR, 2)

        plate_roi = frame[y1:y2, x1:x2]
        if plate_roi.size == 0:
            continue

        processed_plate = preprocess_plate(plate_roi)
        try:
            ocr_results = ocr_reader.readtext(processed_plate,
                                             allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        except Exception as e:
            print(f"Error in OCR: {e}")
            return annotated_frame, None, plate_coords # Return with what we have

        if ocr_results:
            plate_text = " ".join(res[1] for res in ocr_results)  # Join multiple OCR detections
            cv2.putText(annotated_frame, plate_text, (x1, y1 - 10),
                        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
            for bbox_info in ocr_results:
                bbox_coords = bbox_info[0]
                x_tl, y_tl = int(bbox_coords[0][0]), int(bbox_coords[0][1])
                x_br, y_br = int(bbox_coords[2][0]), int(bbox_coords[2][1])
                cv2.rectangle(annotated_frame, (x1 + x_tl, y1 + y_tl),
                              (x1 + x_br, y1 + y_br), RED_COLOR, 2)
            return annotated_frame, plate_text, plate_coords # Return first detection

    return annotated_frame, None, None  # No OCR text found

def detect_material_color(frame):
    """
    Detects the predominant color of the material in the frame.  Averages the color
    in the center portion of the image.
    """
    height, width, _ = frame.shape
    center_x = width // 4
    center_y = height // 4
    center_width = width // 2
    center_height = height // 2
    center_crop = frame[center_y:center_y + center_height, center_x:center_x + center_width]

    avg_color = np.mean(center_crop, axis=(0, 1))
    b, g, r = avg_color

    if r > 120 and r > g * 1.2 and r > b * 1.2:  # More robust thresholds
        return "Red Material"
    elif g > 120 and g > r * 1.2 and g > b * 1.2:
        return "Green Material"
    elif b > 120 and b > r * 1.2 and b > g * 1.2:
        return "Blue Material"
    else:
        return "Unknown Material"

# --------------------------------------------------
# Output Functions
# --------------------------------------------------
def copy_and_notify(text):
    """Copies text to the clipboard and displays a notification."""
    try:
        pyperclip.copy(text)
    except Exception as e:
        print(f"Error copying to clipboard: {e}")
    try:
        notification.notify(
            title="Number Plate Detected",
            message=text,
            timeout=4
        )
    except Exception as e:
        print(f"Error displaying notification: {e}")

# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():
    """
    Main loop to:
    1. Capture video from stream.
    2. Detect plate and read text.
    3. Detect material color.
    4. Display results.
    """
    yolo_model, ocr_reader = load_models()
    if yolo_model is None or ocr_reader is None:
        print("Failed to load models. Exiting.")
        return

    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Error: Unable to open stream")
        return

    last_processed_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        if time.time() - last_processed_time > TIME_BETWEEN_FRAMES:
            annotated_frame, plate_text, plate_coords = detect_plate_and_text(
                frame, yolo_model, ocr_reader)
            if plate_text:
                print(f"[INFO] Plate: {plate_text}, Coordinates: {plate_coords}")
                copy_and_notify(plate_text)
                material_color = detect_material_color(frame)
                print(f"[INFO] Material: {material_color}")
            else:
                print("[INFO] No plate found.")
            last_processed_time = time.time()
        else:
            annotated_frame = frame

        cv2.imshow("ANPR and Material Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


