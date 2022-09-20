import cv2
import glob 
from skimage.filters import gaussian
from skimage import img_as_ubyte

path = "E:/Drishti/Dataset/*.*"
img_number = 1

for file in glob.glob(path):
    print(file)
    img = cv2.imread(file, 0)
    
    smoothed_image = img_as_ubyte(gaussian(img, sigma=5, mode='constant', cval=0.0))
    
    cv2.imwrite("E:/Drishti/smooth/smoothed_image"+str(img_number)+".png", smoothed_image)
    img_number +=1    
    

#Capture all mages into an array and then iterate through each image
#Normally used for machine learning workflows.

import numpy as np
import cv2
import os
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte

images_list = []
SIZE = 512

path = "E:/Drishti/Dataset/*.*"

#First create a stack array of all images
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 0)  #now, we can read each file since we have the full path
    img = cv2.resize(img, (SIZE, SIZE))
    images_list.append(img)
        
images_list = np.array(images_list)

#Process each slice in the stack
img_number = 1
for image in range(images_list.shape[0]):
    input_img = images_list[image,:,:]  #Grey images. For color add another dim.
    smoothed_image = img_as_ubyte(gaussian(input_img, sigma=5, mode='constant', cval=0.0))
    cv2.imwrite("E:/Drishti/smooth/smoothed_image"+str(img_number)+".png", smoothed_image)
    img_number +=1     
   
    