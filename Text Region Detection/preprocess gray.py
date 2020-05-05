from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

def Image_Pro(Im_Seq):

   
   Im_Con_Gray = cv2.cvtColor(Im_Seq, cv2.COLOR_BGR2GRAY)
   
   kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
   
   Im_Grad = cv2.morphologyEx(Im_Con_Gray, cv2.MORPH_GRADIENT, kernel1) 
   
   filename = "gray scale modified %d.jpg" %j
   cv2.imwrite(filename, Im_Grad)


         




mypath='E:/rough text data set/non text region'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]))


for j in range(1, len(images)):
   Image_Pro(images[j])

