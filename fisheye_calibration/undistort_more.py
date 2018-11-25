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

balance=1.5
dim2=None
dim3=None
DIM= (640, 480)
K = np.array([[392.2587784103325, 0.0, 349.5946215390046], [0.0, 429.2697800524658, 219.08594079864957], [0.0, 0.0, 1.0]])
D= np.array([[-0.5947515245760245], [1.210686515239806], [-1.8039526380719182], [0.7464215045731002]])

img = cv2.imread('cal2.jpg',0)

dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

if not dim2:
    dim2 = dim1

if not dim3:
    dim3 = dim1

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

#Calcula mapas que resuelve la distorsión y rectifica con remap()
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
#Transforma la imagen a quitar efecto ojo de pez
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("Distorted", img)
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

