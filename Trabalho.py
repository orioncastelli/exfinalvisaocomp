# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:14:02 2018

@author: 
    Bruno Hjort
    Emanoel Kruger
    Orion Silva
"""

import cv2
from keras import models
from tensorflow.keras.preprocessing import image
#import numpy

cap = cv2.VideoCapture(0)
cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

base_dir = "G:\\Pós\\Visão Computacional\\Aula 4\\Treinamento3"
model = models.load_model(base_dir+"\\Humor_Emanoel.h5")

classes=['Brabo','Feliz','Neutro']

while(True):

	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	#print(faces)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

		rosto = img[y:y+h,x:x+w]
		rosto = cv2.resize(rosto, (100, 100))
		rosto = image.img_to_array(rosto)
		rosto = rosto.reshape((1,) + rosto.shape)
    
		predict = model.predict_classes(rosto)
		print(classes[predict[0]])
		cv2.putText(img, classes[predict[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
		print('Círculo')

	cv2.imshow('Image',img)

	if cv2.waitKey(1) == 27:
		break

cap.release()
cv2.destroyWindow('Image')
cv2.destroyAllWindows()