# Autor: Kenneth Jiang
# Post: Calibrate fishete lens using OpenCV
# Obtenido en: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# Publicado: 28/09/2017
#
# Mejora: leer variables DIM, K y D de un archivo generadas por código de calibración
#
# You should replace these 3 lines with the output in calibration step

variables=open("variables_fisheye.txt")
DIM=variables.readline().split()[1]
K=np.array( variables.readline().split()[1] )
D=np.array( variables.readline().split()[1] )

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    #Calcula mapas que resuelve la distorsión y rectifica con remap()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    #Transforma la imagen a quitar efecto ojo de pez
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("Imagen con distorsión", img)
    cv2.imshow("Imagen sin distorsión", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)

variables.close()