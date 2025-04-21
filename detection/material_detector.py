import numpy as np

def detect_material_color(frame):
    avg_color = np.mean(frame, axis=(0, 1))  # BGR format
    b, g, r = avg_color

    if r > 150 and r > g and r > b:
        return "Red Material"
    elif g > 150 and g > r and g > b:
        return "Green Material"
    elif b > 150:
        return "Blue Material"
    else:
        return "Unknown Material"