

#Third attempt
import threading

import cv2
import numpy as np
import pytesseract
import easyocr
import warnings
import mysql.connector
import tkinter as tk
from threading import Thread
import os
import queue
import _thread as thread
import time
from datetime import datetime, timedelta
import datetime
#from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
#pytesseract.pytesseract.tesseract_cmd= 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#import easyocr

mydb = mysql.connector.connect(host="localhost",user="root",passwd="",database="fyp")
mycursor= mydb.cursor() # to point at database table



net = cv2.dnn.readNet("Resources/yolov4-tiny-custom_last.weights", "Resources/yolov4-tiny-custom .cfg")
classes = ['Platenum']
cap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'], gpu=False)  # what the reader expect from  the image
global q
 #start time





def OCR(cropped_image):

    start = time.time()

    result = reader.readtext(cropped_image)
    text = ''
    for result in result:
        text += result[1] + ' '

    spliced = (remove(text)).upper()

    if(len(spliced) > 5) :
        print(spliced)
        mycursor.execute("SELECT * FROM registered_vehicle WHERE lp = %s", (spliced,))
        exist = mycursor.fetchone()
        # root = tk.Tk()
        # root.title("barrier gate status")

        if not exist:
            print("Record does not exist")
            print("gate-closing........")
            status = 0
            carway2='3'


            # tk.Label(root, text="Barrier stay closed").pack()
            # root.after(10000, lambda: root.destroy())
            #
            # root.mainloop()

        else:
            print("Record found")
            print("gate-opening.........")
            status = 1
            # tk.Label(root, text="Barrier gate opened").pack()
            # root.after(10000, lambda: root.destroy())
            #
            # root.mainloop()

            # mycursor.execute("SELECT * FROM barriergate WHERE lp = %s", (spliced,))
            # exist = mycursor.fetchone()

            mycursor.execute("SELECT  * FROM barriergate WHERE lp = %s ORDER BY bg_id DESC LIMIT 1", (spliced,))
            newresult = mycursor.fetchall()


            if not newresult:
                carway2='1'


            else:

                for x in newresult:
                    pass

                carway=x[4]
                print("carway = ", carway)
                if carway == '0':
                    carway2='1'


                else:
                    carway2='0'



        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        end = time.time()
        diff = end - start


        try:
            queries = "INSERT INTO barriergate(lp,timestamp,status,carway) values (%s, %s , %s, %s)"
            mycursor.execute(queries, (spliced, timestamp, status, carway2))
            mydb.commit()
            print(mycursor.rowcount, "record updated successfully")
        except:
            mydb.rollback()
            print("record fail to update")



    else:
        print(" License plate is blurry / not clear please capture again")
def remove(string):
    return "".join(string.split())

def detection():

    status = 0
    while 1:
        cv2.waitKey(1)
        #_, pre_img = cap.read()
        #pre_img= cv2.resize(pre_img, (640, 480))
        _, img = cap.read()
        #img = cv2.flip(pre_img,1)
        hight, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_name = net.getUnconnectedOutLayersNames()

        layerOutputs = net.forward(output_layers_name)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3] * hight)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                detected_image = img[y:y+h, x:x+w]
                cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
                #print(detected_image)
                cv2.imshow('detection',detected_image)
                cv2.imwrite('lp5.jpg',detected_image)
                cropped_image = cv2.imread('lp5.jpg')

                end_time = time.time()
                diff = end_time - start_time




                if diff > 5:

                    return cropped_image
                    break

        cv2.imshow('img', img)





while 1:
    start_time = time.time()
    old_time = time.time()

    pre_result= detection()

    OCR(pre_result)

    new_time = time.time()
    difference = new_time - old_time


    if difference > 5:
        print("one cycle complete")
        print("-----------------------------")



# Database
# reader = easyocr.Reader(['en'] , gpu=False)  # what the reader expect from  the image
# result = reader.readtext(cropped_image)
# text = ''
# for result in result:
#     text += result[1] + ' '
#
# spliced = (remove(text))
# print('Licese Plate = ',remove(text))
# mycursor.execute("SELECT * FROM dummy WHERE lp = %s", (spliced,))
# exist = mycursor.fetchone()
# root = tk.Tk()
# root.title("barrier gate status")
# cap.release()
# cv2.destroyAllWindows()
# if not exist:
#     print("Record does not exist")
#
#
#     tk.Label(root, text="Barrier stay closed").pack()
#     root.after(10000, lambda: root.destroy())
#
#     root.mainloop()
#
# else:
#     print("Record found")
#     tk.Label(root, text="Barrier gate opened").pack()
#     root.after(10000, lambda: root.destroy())
#
#     root.mainloop()

# queries = "INSERT INTO dummy_lp(id,lp) VALUES (%s, %s)"
# mycursor.execute(queries,spliced)
#
# mydb.commit()
# print(mycursor.rowcount, "record updated successfully")




