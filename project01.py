import cv2
import os
#print(cv2.__version__)
from matplotlib import pyplot as plt
import shutil
source_path = os.path.join("C:\\", "Users", "sadhu", "OneDrive", "Pictures", "Screenshots", "Screenshot 2024-01-05 154535.png")
destination_path = "E:\opencv projects\openCV projects\opencvprojects\images"
shutil.copy(source_path,destination_path)
image = cv2.imread("E:\opencv projects\openCV projects\opencvprojects\images\Screenshot 2024-01-05 154535.png")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
