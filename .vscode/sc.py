import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from image_processing import bgr_to_rgb
from number_detection import detect_numbers
from line_detection import *
from number_classification import *
from keras.models import load_model

video_directory = 'data/videos/'
cnn_path = 'cnn.h5'

if os.path.isfile(cnn_path): 
    #ucitavanje CNN iz fajla (ako postoji)
    cnn = load_model('cnn.h5')
else:
    #treniranje CNN
    cnn = train_cnn()

#ucitavanje videa iz "data/videos"
for video_name in os.listdir(video_directory):

    video_path = os.path.join(video_directory, video_name)
    print("Processing video -> " + video_name)

    cap = cv2.VideoCapture(video_path)
    

    if (cap.isOpened() == False): 
        print("Error opening video from file")

    """ #write video to file
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height)) """
    
    #citanje do kraja videa
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            #konverzija u RGB
            image = bgr_to_rgb(frame)

            image_show = image.copy()

            #detekcija plave linije
            lower_blue = np.array([110,50,20])
            upper_blue = np.array([130,255,255])
            
            #detect_line(image_mask(lower_blue, upper_blue, image), image_show)

            #detekcija zelene linije
            lower_green = np.array([55,50,20])
            upper_green = np.array([65,255,255])
            
            #detect_line(image_mask(lower_green, upper_green, image), image_show)

            #detekcija brojeva
            numbers_array, numbers_coordinates = detect_numbers(image, image_show)
            
            i = 0
            for number in numbers_array:               
                pic =  number.reshape(1,28,28,1)

                #predikcija vrednosti broja
                result = classification(pic, cnn)

                #cv2.imshow('nse',number)

                #stampanje broja na videu
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_show,str(result),(numbers_coordinates[i][0]+2,numbers_coordinates[i][1]+2), font, 1,(139,0,139),2,cv2.LINE_AA)

                i += 1 

            #prikazivanje videa -> before & after
            cv2.imshow(video_name + " new image", image_show)
            #cv2.imshow(video_name + " original image", image)
            """ out.write(image_show) """

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break        

    cap.release()   
    cv2.destroyAllWindows()

    break #zaustavljanje posle prvog videa

