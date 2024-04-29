import cv2
import easyocr
import numpy as np

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Open the default camera (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames and perform text detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is captured successfully
    if ret:
        # Detect text in the frame
        results = reader.readtext(frame)

        # Display the detected text on the frame
        for (bbox, text, _) in results:
            # Convert the bounding box coordinates to integer values
            bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]

            # Reshape the bounding box to match the format expected by cv2.polylines
            pts = np.array(bbox)

            # Draw the bounding box on the frame
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Display the OCR'd text on the frame
            cv2.putText(frame, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with text detection
        cv2.imshow('Real-time Text Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
