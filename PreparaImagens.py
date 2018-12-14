# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:20:46 2018

@author: Bruno Hjort, Órion Silva, Emanoel Krueger
"""

import cv2
import os

base_dirs = ["./Treinamento2/Treina/Brabo", "./Treinamento2/Treina/Neutro", "./Treinamento2/Treina/Feliz","./Treinamento2/Valida/Brabo", "./Treinamento2/Valida/Neutro", "./Treinamento2/Valida/Feliz"]
dest_dirs = ["./Treinamento3/Treina/Brabo", "./Treinamento3/Treina/Neutro", "./Treinamento3/Treina/Feliz","./Treinamento3/Valida/Brabo", "./Treinamento3/Valida/Neutro", "./Treinamento3/Valida/Feliz"]

for i in range(6):
	base_dir = base_dirs[i]
	dest_dir = dest_dirs[i]
	print("Lendo", base_dir)
	for root, dirs, files in os.walk(base_dir):
		for filename in files:
			img = cv2.imread(base_dir + '/' + filename)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(gray, (100, 100))	
			cv2.imwrite(dest_dir + '/' + filename, resized)				
			
