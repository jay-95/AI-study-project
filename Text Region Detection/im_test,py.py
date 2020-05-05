import cv2
import numpy as np


#이미지 입력 및 그레이 변환
Im_Contour = cv2.imread('d2 image 1.png');
Im_Con_Gray = cv2.cvtColor(Im_Contour, cv2.COLOR_BGR2GRAY);

Im_Clone1 = Im_Contour.copy()
Im_Clone2 = Im_Contour.copy()

#Morph 그래디언트
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
kernel3 = np.ones((11,11), np.uint8)
Im_Grad = cv2.morphologyEx(Im_Con_Gray, cv2.MORPH_GRADIENT, kernel1)


#adaptive mean
Im_Th1 = cv2.adaptiveThreshold(Im_Grad, 225, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3,12);

#adaptive gaussian
Im_Th2 = cv2.adaptiveThreshold(Im_Grad, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3,12);

closing = cv2.morphologyEx(Im_Th2, cv2.MORPH_CLOSE,kernel2)

contours, hierachy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(Im_Clone1, contours, -1, (0,255,0), 1)

i = 1

for cnt in contours:
   x, y, w, h = cv2.boundingRect(cnt)
   if (h > 10 and w > 40) and not(w >= 512 - 5 and h >= 512 - 5):
     cv2.rectangle(Im_Clone2, (x, y), (x+w,y+h), (0,0,255), 1)
     crop = Im_Clone1[y:y+h, x:x+w]
     filename = "%d" %i
     cv2.imshow(filename, crop)
     i+=1



cv2.imshow('Mean Adaptive Threshold', Im_Th1)
cv2.imshow('Gaussian Adaptive Threshold', Im_Th2)
cv2.imshow('Gradient', Im_Grad)
cv2.imshow('Close', closing)
cv2.imshow('Contour 1', Im_Clone1)
cv2.imshow('Contour 2', Im_Clone2)

cv2.waitKey(0)
cv2.destroyAllWindows()