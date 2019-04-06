import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_processing import *

def detect_numbers(image):
    #konverzija u grayscale
    grayscale_image = rgb_to_grayscale(image)

    #median blur za smanjenje suma
    image_blur = blur(grayscale_image)

    #otvaranje -> erozija + dilacija za uklanjanje suma
    image_opening = dilate_image(erode_image(image_blur))

    #konverzija u binarnu sliku
    image_processed = grayscale_to_binary(image_opening)

    #pronalazenje kontura
    contours_image, contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    numbers_array = []
    coordinates_array = []

    i = 0
    for contour in contours: 
        #koordinate pravougaonika
        x,y,w,h = cv2.boundingRect(contour)

        x -= 2
        y -= 2
        w += 2
        h += 2
  
        #povrsina konture
        area = cv2.contourArea(contour) 

        #cuvanje pronadjenog regiona
        if area < 500 and h > 10 and hierarchy[0][i][3] == -1:        
            number = grayscale_image[y:y+h,x:x+w]

            #promena velicine u 28x28 + skaliranje u [0,1]
            number = scale_to_range(resize_image(number))

            numbers_array.append(number)
            coordinates_array.append([x,y,h,w])

        i += 1

    return numbers_array, coordinates_array