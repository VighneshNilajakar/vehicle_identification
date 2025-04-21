import easyocr

reader = easyocr.Reader(['en'])

def detect_number_plate(frame):
    results = reader.readtext(frame)
    for (_, text, prob) in results:
        if len(text) >= 6 and prob > 0.5:
            return text
    return "Number plate not detected"