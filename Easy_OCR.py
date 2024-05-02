import cv2
import easyocr
import numpy as np
import threading

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Function for text detection on frames
def detect_text(frame):
    results = reader.readtext(frame)
    for (bbox, text, _) in results:
        bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]
        pts = np.array(bbox)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        cv2.putText(frame, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Real-time Text Detection', frame)

# Function for capturing frames and performing text detection
def process_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_text(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Start the frame processing thread
thread = threading.Thread(target=process_frames)
thread.start()
