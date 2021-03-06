# -*- coding: utf-8 -*-
#
# Autor: Kenneth Jiang
# Post: Calibrate fishete lens using OpenCV
# Obtenido en: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# Publicado: 28/09/2017
#
# Mejora: guardar variables DIM, K y D en un archivo para ser utilizadas en otro programa

import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (3,3)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
frame_gray = None

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
cont =0
images = glob.glob('*.jpg')
for fname in images:
    
    img = cv2.imread(fname)
    rows, cols = img.shape[:-1]
    # rota la imagen 90ª en sentido antihoraio
    m = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
    img = cv2.warpAffine(img, m, (cols, rows))
    if _img_shape == None:
		_img_shape = img.shape[:2]
    else:
		assert _img_shape == img.shape[:2], "All images must share the same size."
    frame_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(frame_gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
	    objpoints.append(objp)
	    cv2.cornerSubPix(frame_gray,corners,(3,3),(-1,-1),subpix_criteria)
	    imgpoints.append(corners)
	    cv2.imshow("Valid Images", frame_gray )
    cv2.waitKey(1)
    cont = cont + 1
    print(cont)

cv2.waitKey(0)
cv2.destroyAllWindows()
N_OK = len(objpoints)
# Output floating-point camera matrix
K = np.zeros((3, 3))
# Output vector of distortion coefficients
D = np.zeros((4, 1))
#output vector of rotation vectors
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
#output vector of translation vectors
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
	objpoints,
	imgpoints,
	frame_gray.shape[::-1], #image size
	K,
	D,
	rvecs,
	tvecs,
	calibration_flags,
	(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

DIM = str(_img_shape[::-1])
K = str(K.tolist())
D = str(D.tolist())

print("Found " + str(N_OK) + " valid images for calibration\n")
print("DIM=" + DIM + "\n")
print("K=np.array(" + K + ")\n")
print("D=np.array(" + D + ")\n")

archivo = open("variables_fisheye.txt","w")
archivo.write("DIM=" + DIM + "\n" +
		  "K=" + K + "\n" +
		  "D=" +D)
archivo.close()








