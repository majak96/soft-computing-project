import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from image_processing import bgr_to_rgb, image_mask, rgb_to_bgr
from number_detection import detect_numbers
from line_detection import detect_line
from number_tracking import *
from number_classification import *
from keras.models import load_model  

video_directory = 'data/videos/'
cnn_path = 'cnn.h5'
ann_path = 'ann.h5'

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
    frame_counter = 0

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

    #citanje do kraja videa
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        found_in_frame = []

        if ret == True:
            
            #konverzija u RGB
            image = bgr_to_rgb(frame)

            image_show = image.copy()

            #detekcija linija u prvom frejmu
            if lines_detected == False: 
                #detekcija plave linije
                lower_blue = np.array([110,50,20])
                upper_blue = np.array([130,255,255])
                
                blue_a, blue_b = detect_line(image_mask(lower_blue, upper_blue, image))

                #detekcija zelene linije
                lower_green = np.array([55,50,20])
                upper_green = np.array([65,255,255])
                
                green_a, green_b = detect_line(image_mask(lower_green, upper_green, image))

                lines_detected = True

            #crtanje linija
            cv2.line(image_show,blue_a,blue_b,(255,255,0),1)
            cv2.line(image_show,green_a,green_b,(255,255,0),1)

            #detekcija brojeva
            numbers_array, numbers_coordinates = detect_numbers(image)
                       
            for i in range(len(numbers_array)):
                x,y,h,w = numbers_coordinates[i][0], numbers_coordinates[i][1], numbers_coordinates[i][2], numbers_coordinates[i][3]

                """pic =  numbers_array[i].reshape(1,28,28,1)

                #predikcija vrednosti broja
                result = classification(pic, cnn)

                #stampanje broja na videu
                cv2.putText(image_show,str(result),(x+w+2,y-2),cv2.FONT_HERSHEY_PLAIN,2,(139,0,139),1,cv2.LINE_AA)

                #oznacavanje pronadjene konture
                cv2.rectangle(image_show,(x,y),(x+w,y+h),(139,0,139),1)"""
                
                #centar konture
                centerX = int(x + w/2)
                centerY = int(y + h/2)

                #provera da li je broj vec pronadjen
                found_number = check_if_number_exists(detected_numbers_array, centerX, centerY)

                #ukoliko broj jos uvek nije pronadjen
                if found_number == None:
                    value = classification(numbers_array[i].reshape(1,28,28,1), cnn)
                    
                    dn = DetectedNumber(centerX, centerY, id, value, [x,y,h,w], h*w)
                    detected_numbers_array.append(dn)
                    found_in_frame.append(dn)

                    id += 1               
                else: #ukoliko je broj vec prethodno pronadjen, update koordinata njegovog centra
                    found_number.update_coordinates(centerX, centerY, [x,y,h,w])
                    found_in_frame.append(dn)

                    #ako je nestao, a sad je pronadjen
                    if found_number.not_found_frames != 0:
                        number.not_found_frames = 0

                    """ if frame_counter % 20 == 0:
                        area = h*w

                        if area > found_number.area:
                            value = classification(numbers_array[i].reshape(1,28,28,1), cnn)

                            found_number.value = value

                        found_number.area = area """
                    
            #provera da li neki od detektovanih brojeva prolazi plavu liniju
            for number in detected_numbers_array:
                point = (number.coordinates[0]+number.coordinates[3], number.coordinates[1]+number.coordinates[2])
                                
                if distance_from_line(point, blue_a, blue_b) < 10 and number.coordinates[1]+number.coordinates[2] < max(blue_a[1], blue_b[1]) and number.coordinates[1]+number.coordinates[2] > min(blue_a[1], blue_b[1]):
                    if check_if_next_to_line(number, blue_a, blue_b) and check_if_id_exists(blue_line_crossed, number.id) == False:
                            blue_line_crossed.append(number)
                            #print("+ " + str(number.value))

            #provera da li neki od detektovanih brojeva prolazi zelenu liniju
            for number in detected_numbers_array:
                point = (number.coordinates[0]+number.coordinates[3], number.coordinates[1]+number.coordinates[2])
                
                if distance_from_line(point, green_a, green_b) < 10 and number.coordinates[1]+number.coordinates[2] < max(green_a[1], green_b[1]) and number.coordinates[1]+number.coordinates[2] > min(green_a[1], green_b[1]):
                    if check_if_next_to_line(number, green_a, green_b) and check_if_id_exists(green_line_crossed, number.id) == False:
                            green_line_crossed.append(number)
                            #print("- " + str(number.value))

            #proverava da li je neki od brojeva nestao u ovom frejmu
            for number in detected_numbers_array:
                if number not in found_in_frame:
                    number.not_found_frames += 1

                    #ukoliko ga nema duze vreme u videu - moze da se obrise
                    if number.not_found_frames > 500:
                        detected_numbers_array.remove(number)
            
            #prikazivanje videa
            #cv2.imshow(video_name, rgb_to_bgr(image_show))
            #cv2.imshow(video_name + "processed", new_image)
            
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
    print("Result: " + str(suma))

    #break #zaustavljanje posle prvog videa

results_file.close()