import cv2
from config import CAMERA_URL

def get_camera_stream():
    cap = cv2.VideoCapture(CAMERA_URL)
    return cap