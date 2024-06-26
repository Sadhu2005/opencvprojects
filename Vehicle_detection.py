import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# Create our video capturing object
cap = cv2.VideoCapture('Video/cars.mp4')

# Load our vehicle classifier
vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

# Read first frame
ret, frame = cap.read()

# Ret is True if successfully read
if ret:

    # Grayscale our image for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    vehicles = vehicle_detector.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Release our video capture
cap.release()
imshow("Vehicle Detector", frame)
#cv2.imwrite('Vehicle Detector.jpg',frame)



# Get the height and width of the frame (required to be an interfer)
w = int(cap.get(3))
h = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('cars_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

# Loop once video is successfully loaded
while (True):

    ret, frame = cap.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our body classifier
        vehicles = vehicle_detector.detectMultiScale(gray, 1.2, 3)

        # Extract bounding boxes for any bodies identified
        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Write the frame into the file 'output.avi'
        out.write(frame)
    else:
        break

cap.release()
out.release()