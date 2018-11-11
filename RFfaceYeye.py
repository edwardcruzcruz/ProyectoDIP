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
cap = cv2.VideoCapture(0)
while(True):
	cv2.namedWindow('Ventana', cv2.WINDOW_AUTOSIZE)
	cv2.namedWindow('VentanaROI', cv2.WINDOW_AUTOSIZE)
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
    		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    		#roi_gray = gray[y:y+h, x:x+w]
    		roi_color = img[y:y+h, x:x+w]
		face_resize = cv2.resize(roi_color, (img_width, img_height))		
		cv2.imshow('VentanaROI',face_resize)
		if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
			out=cv2.imwrite('att_faces/orl_faces/'+folder+'/img'+str(img_counter)+'.png',face_resize)
			img_counter=img_counter+1			
			print("Imagen guardada")	
		

	cv2.imshow('Ventana',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

cap.release()
cv2.destroyAllWindows()
