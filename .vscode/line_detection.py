import matplotlib.pyplot as plt
import numpy as np
import cv2
from statistics import mean
from math import hypot

from image_processing import rgb_to_grayscale

def detect_line(image):

    #konverzija u grayscale
    grayscale_image = rgb_to_grayscale(image)

    #Canny Edge Detection
    edges_image = cv2.Canny(image, 50, 150, 3)

    #detekcija linija pomocu Hoguh transformacije
    lines = cv2.HoughLinesP(edges_image, 1, np.pi/180, 50, None, 100, 5)

    #trazenje najduze linije
    point_a = (lines[0][0][0],lines[0][0][1])
    point_b = (lines[0][0][2],lines[0][0][3])

    for line in lines:
        for x1,y1,x2,y2 in line:

            if hypot(x2-x1, y2-y1) > hypot(point_b[0]-point_a[0], point_b[1]-point_a[1]):
                point_a = (x1,y1)
                point_b = (x2,y2)
    
    return point_a, point_b