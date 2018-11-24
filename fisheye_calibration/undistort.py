# -*- coding: utf-8 -*-
#
# Autor: Kenneth Jiang
# Post: Calibrate fishete lens using OpenCV
# Obtenido en: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# Publicado: 28/09/2017
#
# You should replace these 3 lines with the output in calibration step

import sys
import cv2
import numpy as np

variables=open("variables_fisheye.txt")
DIM= (640, 480)
K = np.array([[157.80318621083393, 0.0, 231.84078724271788], [0.0, 147.79333039625973, 72.28830095657395], [0.0, 0.0, 1.0]])
D= np.array([[2.0685142995515053], [-5.079635173054507], [7.018676629889146], [-3.655967741240021]])

img = cv2.imread('cal2.jpg',1)
h,w = img.shape[:2]
#Calcula mapas que resuelve la distorsión y rectifica con remap()
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#Transforma la imagen a quitar efecto ojo de pez
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("Distorted", img)
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

