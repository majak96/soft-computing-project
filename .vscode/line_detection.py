import matplotlib.pyplot as plt
import numpy as np
import cv2
from statistics import mean

from image_processing import *

def detect_line(image, image_show):

    #konverzija u grayscale
    grayscale_image = rgb_to_grayscale(image)

    #Canny Edge Detection
    edges_image = cv2.Canny(image,50,150,3)

    #detekcija linija pomocu Hoguh transformacije
    lines = cv2.HoughLinesP(edges_image,1,np.pi/180,50,None,100,5)

    x1_list = []; x2_list = []; y1_list = []; y2_list = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            #crtanje linije na slici
            #cv2.line(image_show,(x1,y1),(x2,y2),(139,0,139),2)
            x1_list.append(x1)
            x2_list.append(x2)
            y1_list.append(y1)
            y2_list.append(y2)

    first = (mean(x1_list), mean(y1_list))
    second = (mean(x2_list), mean(y2_list))

    #cv2.line(image_show,first,second,(139,0,139),2)
    
    #coefficients = np.polyfit(first, second, 1)

    return first, second