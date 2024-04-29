from matplotlib import pyplot as plt
import cv2
import easyocr
import time
import numpy as np

# Function to display images
import matplotlib.pyplot as plt

def imshow(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


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
        ts = time.time()
        results = reader.readtext(frame)
        te = time.time()
        td = te - ts

        # Display the detected text on the frame
        # Iterate over the detected text results
        # Iterate over the detected text results
        # Iterate over the detected text results
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
        imshow('Real-time Text Detection', frame)

        # Print processing time
        print(f'Processing time: {td:.2f} seconds')

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
