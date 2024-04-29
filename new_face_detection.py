import cv2
import numpy as np
from matplotlib import pyplot as plt
def imshow(title = "Image",image = None,size = 10):
    w, h = image.shape[0],image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
def take_photo():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Press Space to Capture', frame)

        # Check if the user pressed the spacebar key
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar key
            # Save the frame as an image file
            cv2.imwrite("captured_photo.jpg", frame)
            print("Photo captured and saved as captured_photo.jpg")
            break
        elif key == 27:  # Escape key
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

take_photo()

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

img = cv2.imread('captured_photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow('Original image',gray)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No Face Found")

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

imshow('Eye & Face Detection',img)
cv2.imwrite('face_eye.jpg',img)
