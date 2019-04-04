import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from image_processing import bgr_to_rgb, image_mask
from number_detection import detect_numbers
from line_detection import detect_line
from number_tracking import *
from number_classification import *
from keras.models import load_model  

video_directory = 'data/videos/'
cnn_path = 'cnn.h5'

results_file = open("out.txt","w+")
results_file.write("RA 16/2015 Marijana Kološnjaji\r")
results_file.write("file\tsum\r")

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
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height)) 
    """
    
    #za detekciju linija
    green_a = 0; green_b = 0; blue_a = 0; blue_b = 0
    lines_detected = False

    #za pracenje objekata
    id = 0
    detected_numbers_array = [] #svi detektovani brojevi
    blue_line_crossed = [] #brojevi koji su presli plavu liniju
    green_line_crossed = [] #brojevi koji su presli zelenu liniju

    frame_counter=0
    #citanje do kraja videa
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            
            #konverzija u RGB
            image = bgr_to_rgb(frame)

            image_show = image.copy()

            #detekcija linija u prvom frejmu
            if lines_detected == False: 
                #detekcija plave linije
                lower_blue = np.array([110,50,20])
                upper_blue = np.array([130,255,255])
                
                blue_a, blue_b = detect_line(image_mask(lower_blue, upper_blue, image), image_show)

                #detekcija zelene linije
                lower_green = np.array([55,50,20])
                upper_green = np.array([65,255,255])
                
                green_a, green_b = detect_line(image_mask(lower_green, upper_green, image), image_show)

                lines_detected = True

            #detekcija brojeva
            numbers_array, numbers_coordinates = detect_numbers(image, image_show)
                       
            for i in range(len(numbers_array)):               
                pic =  numbers_array[i].reshape(1,28,28,1)

                """ #predikcija vrednosti broja
                result = classification(pic, cnn)

                #cv2.imshow('nse',number)

                #stampanje broja na videu
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_show,str(result),(numbers_coordinates[i][0]+2,numbers_coordinates[i][1]+2), font, 1,(139,0,139),2,cv2.LINE_AA)  """
                
                x,y,h,w = numbers_coordinates[i][0], numbers_coordinates[i][1], numbers_coordinates[i][2], numbers_coordinates[i][3]

                #centar konture
                centerX = int(x + w/2)
                centerY = int(y + h/2)

                #provera da li je broj vec pronadjen
                found_number = check_if_number_exists(detected_numbers_array, centerX, centerY)

                #ukoliko broj jos uvek nije pronadjen
                if found_number == None:
                    value = classification(numbers_array[i].reshape(1,28,28,1), cnn)
                    
                    dn = DetectedNumber(centerX,centerY,id,value,h)
                    detected_numbers_array.append(dn)

                    id += 1               
                else: #ukoliko je broj vec prethodno pronadjen, update koordinata njegovog centra
                    found_number.update_coordinates(centerX,centerY,h)

            """ broj1 = int(max(blue_a[1], blue_b[1])-15)
            broj2 = int(min(blue_a[1], blue_b[1])+15)

            broj3 = int(max(green_a[1], green_b[1])-10)
            broj4 = int(min(green_a[1], green_b[1])+10)

            cv2.line(image_show,(0,broj1),(500,broj1),(255,0,0),5)
            cv2.line(image_show,(0,broj2),(500,broj2),(255,0,0),5)

            cv2.line(image_show,(0,broj3),(500,broj3),(255,0,0),5)
            cv2.line(image_show,(0,broj4),(500,broj4),(255,0,0),5) """
            
            #provera da li je neki od detektovanih brojeva prosao plavu liniju
            for number in detected_numbers_array:
                if distance_from_line(number, blue_a, blue_b) < 20 and number.centerY < max(blue_a[1], blue_b[1])-15 and number.centerY > min(blue_a[1], blue_b[1])+15:
                    if check_if_line_crossed(number, blue_a, blue_b) and check_if_id_exists(blue_line_crossed, number.id) == False:
                            #cv2.circle(image_show,(number.centerX,number.centerY), 10, (255,0,0), -1)
                            blue_line_crossed.append(number)
                            #print("+ " + str(number.value))

            #provera da li je neki od detektovanih brojeva prosao zelenu liniju
            for number in detected_numbers_array:
                if distance_from_line(number, green_a, green_b) < 20 and number.centerY < max(green_a[1], green_b[1])-15 and number.centerY > min(green_a[1], green_b[1])+15:
                    if check_if_line_crossed(number, green_a, green_b) and check_if_id_exists(green_line_crossed, number.id) == False:
                            #cv2.circle(image_show,(number.centerX,number.centerY), 10, (255,0,0), -1)
                            green_line_crossed.append(number)
                            #print("- " + str(number.value))

            #prikazivanje videa -> before & after
            cv2.imshow(video_name + " new image", image_show)
            #cv2.imshow(video_name + " original image", image_processed)
            
            """ out.write(image_show) """

            frame_counter += 1
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

        else:
            break        

    cap.release()   
    cv2.destroyAllWindows()

    suma = calculate_sum(blue_line_crossed, green_line_crossed)
    results_file.write(video_name + "\t" + str(suma) + "\r")
    print("SUM: " + str(suma))

    break #zaustavljanje posle prvog videa

results_file.close()