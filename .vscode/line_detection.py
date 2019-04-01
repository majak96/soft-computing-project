import matplotlib.pyplot as plt
import numpy as np
import cv2

from image_processing import *

def detect_line(image, image_show):

    #konverzija u grayscale
    grayscale_image = rgb_to_grayscale(image)

    #Canny Edge Detection
    edges_image = cv2.Canny(image,50,150,3)

    #detekcija linija pomocu Hoguh transformacije
    lines = cv2.HoughLinesP(edges_image,1,np.pi/180,50,None,100,10)
        
    for line in lines:
        for x1,y1,x2,y2 in line:
            #crtanje linije na slici
            cv2.line(image_show,(x1,y1),(x2,y2),(139,0,139),2)
    
    #TODO: return linije