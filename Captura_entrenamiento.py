# -*- coding: utf-8 -*-
import undistort as ud
import numpy as np
import cv2
import os

img_width, img_height = 92,112
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
folder = raw_input("Ingrese una carpeta de destino: ")
p = os.path.isdir('att_faces/orl_faces/'+folder)
while not p:
	print("Error al ingresar carpeta, ingrese de nuevo la carpeta o cree una en att_faces/orl_faces/<new folder>")
	folder = raw_input("Ingrese una carpeta de destino: ")
	p = os.path.isdir('att_faces/orl_faces/'+folder)


img_counter = 0
cap = cv2.VideoCapture("media/mejorado2.webm")
#cap = cv2.VideoCapture(1)
while(True):
	cv2.namedWindow('Ventana', cv2.WINDOW_AUTOSIZE)
	cv2.namedWindow('VentanaROI', cv2.WINDOW_AUTOSIZE)
	ret, img = cap.read()
	kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
	if ret:
		rows, cols = img.shape[:-1]
		# rota la imagen 90ª en sentido antihoraio
		m = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
		img = cv2.warpAffine(img, m, (cols, rows))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#img = ud.undistort_image(img)    
		sharpened = cv2.filter2D(img, -1, kernel_sharpening)
		cv2.imshow('Image Sharpening', sharpened)    
		#cv2.imshow('nueva',img2)
		faces = face_cascade.detectMultiScale(sharpened, 1.3, 5)
		for (x,y,w,h) in faces:
			sharpened = cv2.rectangle(sharpened,(x,y),(x+w,y+h),(255,0,0),2)
			roi = sharpened[y:y+h, x:x+w]
			face_resize = cv2.resize(roi, (img_width, img_height))		
			cv2.imshow('VentanaROI',face_resize)
			if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
				out=cv2.imwrite('att_faces/orl_faces/'+folder+'/img'+str(img_counter)+'.png',face_resize)
				img_counter=img_counter+1			
				print("Imagen guardada")	
		

		cv2.imshow('Ventana',img)
		cv2.imshow('Image Sharpening', sharpened)
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    	break

cap.release()
cv2.destroyAllWindows()
