#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
#OpenCV trabaja con arreglos de numpy
import numpy as np

#Se importa la lista de personas con acceso al laboratorio
from listaPermitidos import Politecnicos
flabs=Politecnicos()

# Parte 1: Creando el entrenamiento del modelo
print('Formando...')

#Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'att_faces/orl_faces'

#Tamaño para reducir a miniaturas las fotografias
size = 4

# Crear una lista de imagenes y una lista de nombres correspondientes
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dir_faces):
    # walk () genera los nombres de los archivos en un árbol de directorios
    # recorriendo el árbol de arriba hacia abajo o de abajo hacia arriba.
    for subdir in dirs:
        names[id] = subdir
        # une direcciones 'att_faces/orl_faces'+'edward'
        subjectpath = os.path.join(dir_faces, subdir)#une direcciones 'att_faces/orl_faces'+'edward'
        # listdir retorna una lista que contiene los nombres de la entrada de un directorio dado por una ruta
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

# Crear una matriz Numpy de las dos listas anteriores
(images, lables) = [np.array(lis) for lis in [images, lables]]

# OpenCV entrena un modelo a partir de las imagenes
model = cv2.face.LBPHFaceRecognizer_create()
#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createEigenFaceRecognizer()
 
#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()

model.train(images, lables)


# Parte 2: Utilizar el modelo entrenado en funcionamiento con la camara
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('media/anguloSCARACAMARA2.2.webm')

cont=0
while True:
    #leemos un frame lo invertimos 90º y lo guardamos
    rval, frame = cap.read()
    cont=cont+1
    if rval:

        # *** Esto para preprocesar captura de video en cámara MASHI 
        rows, cols = frame.shape[:-1]
        # rota la imagen 90ª en sentido antihoraio
        m = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
        frame = cv2.warpAffine(frame, m, (cols, rows))

        # se incrementa intensidad de brillo
        #frame=cv2.add(frame,50)
        # *** Hasta aquí preprocesamiento por el MASHI

        #invierte la imagen con respecto al eje vertical
        frame=cv2.flip(frame,1,0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #aumentamos el brillo sin desbordar
        #gray= np.where((255-gray)<230, 255, gray*2)
        #gamma=2.0
        #table=np.array([( (i/255.0) ** (1.0/gamma) ) * 255
        #    for i in np.arange(0, 256) ]).astype("uint8")
        #gray=cv2.LUT(gray,table)
        brightness=32
        shadow=brightness
        highlight=255
        alpha=(highlight-shadow)/255
        gamma=shadow
        gray=cv2.addWeighted(gray, alpha, gray, 0, gamma)
        
        #aumentamos el contraste
        contrast=32
        alpha = 131*(contrast + 127)/(127*(131-contrast))
        gamma = 127*(1-alpha)
        gray = cv2.addWeighted(gray, alpha, gray, 0, gamma)
        
        #redimensionar la imagen
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        """buscamos las coordenadas de los rostros (si los hay) y
        guardamos su posicion"""
        faces = face_cascade.detectMultiScale(mini)
        
        for i in range(len(faces)):
            face_i = faces[i]
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            # Intentado reconocer la cara
            prediction = model.predict(face_resize)
            
            #Dibujamos un rectangulo en las coordenadas del rostro
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Escribiendo el nombre de la cara reconocida
            # La variable cara tendra el nombre de la persona reconocida
            cara = '%s' % (names[prediction[0]])

            #Si la prediccion tiene una exactitud menor a 100 se toma como prediccion valida
            if prediction[1]<100:
                #Ponemos el nombre de la persona que se reconoció
                cv2.putText(gray,'%s - %.0f' % (cara,prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

                #En caso de que la cara sea de algun conocido se realizara determinadas accione          
                #Busca si los nombres de las personas reconocidas estan dentro de los que tienen acceso          
                persona=cara.split('_') # Nombre de ejemplo en la lista: 'Bryan Tumbaco'
                flabs.TuSiTuNo(persona[0].capitalize()+' '+persona[1].capitalize())

            #Si la prediccion es mayor a 100 no es un reconomiento con la exactitud suficiente
            elif prediction[1]>101 and prediction[1]<500:           
                #Si la cara es desconocida, poner desconocido
                cv2.putText(gray, 'Desconocido',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))  

            #Mostramos la imagen
            cv2.imshow('OpenCV Reconocimiento facial', gray)

    #Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
