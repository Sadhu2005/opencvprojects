import cv2
import numpy as np
from matplotlib import pyplot as plt
import shutil
import os
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
source_path = os.path.join("C:\\","Users","sadhu","OneDrive","Pictures","Screenshots","Screenshot 2024-03-21 194111.png")
destination_path = "E:\opencv projects\openCV projects\opencvprojects\images"
shutil.copy(source_path,destination_path)
image = cv2.imread('E:\opencv projects\openCV projects\opencvprojects\images\Screenshot 2024-03-21 194111.png')
#imshow('God Shiva',image)
image.shape[:2]  # (1200, 1920) (height, width)


def imshow(title="", image=None, size=10):
    # The line below is changed from w, h to h, w
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w / h

    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


#imshow("God Shiva", image)
# We use cvtColor, to convert to grayscale
# It takes 2 arguments, the first being the input image
# The second being the color space conversion code
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#imshow("Converted to Grayscale", gray_image)
print(image.shape)
print(gray_image.shape)
cv2.imwrite('output.jpg',gray_image)

B, G, R = cv2.split(image)
zeros = np.zeros(image.shape[:2], dtype = "uint8")
imshow("Red", cv2.merge([zeros, zeros, R]))
imshow('blue',cv2.merge([B, zeros,zeros]))
imshow('green',cv2.merge([zeros,G,zeros]))
merged = cv2.merge([B, G, R])
imshow("Merged", merged)
# Let's amplify the blue color
merged = cv2.merge([B+100, G, R])
imshow("Blue Boost", merged)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
imshow('HSV', hsv_image)