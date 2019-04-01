import cv2
import numpy as np

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def grayscale_to_binary(image):
    threshold, image_bin = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
    #print("threshold -> " + str(threshold))

    return image_bin

def blur(image):
    return cv2.medianBlur(image,3)

def dilate_image(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.dilate(image, kernel)

def erode_image(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.erode(image, kernel)

def resize_image(image):
    return cv2.resize(image,(28,28), interpolation = cv2.INTER_NEAREST)

def scale_to_range(image):  
    return image/255

def image_mask(lower_value, upper_value, image):
    #konverzija u HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #kreira masku za odredjenu boju
    mask = cv2.inRange(hsv_image, lower_value, upper_value)
    
    return cv2.bitwise_and(image, image, mask = mask) 