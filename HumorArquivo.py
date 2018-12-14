# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:24:50 2018

@author: Bruno Hjort, Órion Silva, Emanoel Krueger
"""

import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

base_dirs = ["./Treinamento/Treina/Brabo", "./Treinamento/Valida/Brabo", "./Treinamento/Treina/Feliz", 
             "./Treinamento/Valida/Feliz", "./Treinamento/Treina/Neutro", "./Treinamento/Valida/Neutro"]
for base_dir in base_dirs:
	file_idx = 0
	print("Lendo", base_dir)
	for root, dirs, files in os.walk(base_dir):
		for filename in files:
			img = cv2.imread(base_dir + '/' + filename)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)	
			for (x,y,w,h) in faces:
				file_idx = file_idx + 1
				img_face = img[y:y+h,x:x+w]
				new_file = base_dir + '/' + "sep_" + str(file_idx) + "_" + filename
				print(new_file)
				cv2.imwrite(new_file, img_face)