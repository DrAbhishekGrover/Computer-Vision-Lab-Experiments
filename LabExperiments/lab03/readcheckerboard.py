import cv2
import numpy as np
import glob #for file handling
import os # for file file handling

board_size= (11,7)
size_mm=30

#Real world coordinate
objp=np.zeros((board_size[0]*board_size[1],3),np.float32)#matrix which stores grid points for original object
objp[:,:2]=np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
objp=objp*size_mm

#Image coordinates

objpoints=[]
imgpoints=[] #2D points in image plane

#Get all files one by one
img_path=glob.glob('images/checkerboard/cam1/*.png')
#List of file locations with png extension in the directory
if(len(img_path)==0):
    print("no file")

gray_shape2=None
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
for fname in img_path:
    img=cv2.imread(fname)
    gray_shape=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_shape2=np.shape(gray_shape)

    #Find intersecting points in images
    ret,corners=cv2.findChessboardCorners(gray_shape,board_size,None)
    #ret: boolean
    #corners: coordinates for intersections
    if ret:
        objpoints.append(objp)#List of real world coordinates
        corners_refined=cv2.cornerSubPix(gray_shape,
                                         corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners_refined)

cv2.destroyAllWindows()

ret,intrinsic_matrix,dist_coeff,rvecs,tvecs=cv2.calibrateCamera(objpoints,
                                                                imgpoints,
                                                                gray_shape2,
                                                                None,
                                                                None)

print("Intrinsic matrix")
print(intrinsic_matrix)
R,_=cv2.Rodrigues(rvecs[0])#Converts rotation vectors to 3x3 rotation matrix format
#The second returned element in above line is jacobian of R
t=tvecs[0]
extrinsic_matrix=np.hstack((R,t))
print("extrinsic matrix")
print(extrinsic_matrix)
projection_matrix=intrinsic_matrix@extrinsic_matrix
print("projection matrix")
print(projection_matrix)






