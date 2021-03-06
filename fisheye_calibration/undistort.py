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

DIM= (640, 480)
K = np.array([[392.2587784103325, 0.0, 349.5946215390046], [0.0, 429.2697800524658, 219.08594079864957], [0.0, 0.0, 1.0]])
D= np.array([[-0.5947515245760245], [1.210686515239806], [-1.8039526380719182], [0.7464215045731002]])

img = cv2.imread('cal2.jpg',0)
#Calcula mapas que resuelve la distorsión y rectifica con remap()
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
#Transforma la imagen a quitar efecto ojo de pez
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("Distorted", img)
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

