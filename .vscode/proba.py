from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from image_processing import *

""" (X_train, y_train), (X_test, y_test) = mnist.load_data()

i = 0
for img in X_train:
    #threshold, newimg = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)

    mask = img > 0
    coords = np.argwhere(mask)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    cut = img[x0:x1, y0:y1]

    cut = cv2.resize(cut,(28,28), interpolation = cv2.INTER_NEAREST)

    cv2.imwrite("face-" + str(i) + ".png", cut)


    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    i += 1

    if i > 50:
        break """
    

""" video_directory = 'data/videos/'

for video_name in os.listdir(video_directory):

    video_path = os.path.join(video_directory, video_name)
    print("Processing video -> " + video_name)

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False): 
        print("Error opening video from file")
    
    #read until the end
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            #conversion to RGB
            image = bgr_to_rgb(frame)

            #image = white_mask(image)

            cv2.imshow(video_name + " original image", image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break        

    cap.release()   
    cv2.destroyAllWindows()

    break
 """

(X_train, y_train), (X_test, y_test) = mnist.load_data()

i = 0
for img in X_train:

    #conversion to binary image
    image_processed = grayscale_to_binary(img)

    #find contours
    contours_image, contours, hierarchy = cv2.findContours(image_processed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    numbers_array = []

    i = 0
    for contour in contours: 
        #rectangle coordinates
        x,y,w,h = cv2.boundingRect(contour) 
        
        #rectangle area
        area = cv2.contourArea(contour) 

        if h > 10 and hierarchy[0][i][3] == -1:
            #draw rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h),(139,0,139),2)

        i += 1

    #cut = cv2.resize(cut,(28,28), interpolation = cv2.INTER_NEAREST)

    cv2.imshow(" original image", img)


    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    i += 1

    if i > 50:
        break