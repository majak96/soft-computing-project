from math import sqrt
import numpy as np
import cv2

class DetectedNumber:
    def __init__(self, centerX, centerY, id, value, coordinates, area):
        self.centerX = centerX
        self.centerY = centerY
        self.id = id
        self.value = value
        self.coordinates = coordinates
        self.area = area
        self.not_found_frames = 0
      
    def update_coordinates(self, centerX, centerY, coordinates):
        self.centerX = centerX
        self.centerY = centerY
        self.coordinates = coordinates

def check_if_number_exists(detected_numbers, new_center_X, new_center_Y):
    if len(detected_numbers) > 0:

        min_number = detected_numbers[0]
        min_distance = sqrt((min_number.centerX-new_center_X)**2 + (min_number.centerY-new_center_Y)**2)
    
        #pronalazi broj koji je najblizi tom broju
        for number in detected_numbers:
            distance = sqrt((number.centerX-new_center_X)**2 + (number.centerY-new_center_Y)**2)
            
            if distance < min_distance:
                min_distance = distance
                min_number = number
        
        #ako je najblizi broj bio dovoljno blizu - to je isti taj broj
        if min_distance < 5:
            return min_number

    return None

def distance_from_line(point, first_point, second_point):
    x0 = point[0]; y0 = point[1]
    x1 = first_point[0]; y1 = first_point[1]
    x2 = second_point[0]; y2 = second_point[1]

    #racuna se udaljenost od tacke opisane sa (x0,y0) do prave opisane sa (x1,y1) i (x2,y2)
    num1 = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    num2 = sqrt((y2-y1)**2 + (x2-x1)**2)

    return num1/num2

def check_if_id_exists(detected_numbers, id):
    for number in detected_numbers:
        if(number.id == id):
            return True

    return False

def check_if_next_to_line(detected_number, first_point, second_point):
    k = (second_point[1] - first_point[1])/(second_point[0] - first_point[0])
    n = -k*first_point[0] + first_point[1]

    x_parallel = (detected_number.centerY - n)/k

    #provera da li se broj nalazi s leve strane linije
    if x_parallel > detected_number.centerX:
        return True
    else:
        return False

def calculate_sum(blue_line_crossed, green_line_crossed):
    sum = 0

    for number in blue_line_crossed:
        sum += number.value

    for number in green_line_crossed:
        sum -= number.value

    #konacan rezultat
    return sum