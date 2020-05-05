import cv2
import numpy as np


#이미지 입력 및 그레이 변환
Im_Contour = cv2.imread('d2 image 1.png');
Im_Con_Gray = cv2.cvtColor(Im_Contour, cv2.COLOR_BGR2GRAY);

Im_Clone1 = Im_Contour.copy()
Im_Clone2 = Im_Contour.copy()

#Morph 그래디언트
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
kernel3 = np.ones((11,11), np.uint8)
Im_Grad = cv2.morphologyEx(Im_Con_Gray, cv2.MORPH_GRADIENT, kernel1)


#adaptive mean
Im_Th1 = cv2.adaptiveThreshold(Im_Grad, 225, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3,12);

#adaptive gaussian
Im_Th2 = cv2.adaptiveThreshold(Im_Grad, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3,12);

closing = cv2.morphologyEx(Im_Th2, cv2.MORPH_CLOSE,kernel3)

contours, hierachy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(Im_Clone1, contours, -1, (0,255,0), 1)

i = 1

for cnt in contours:
   x, y, w, h = cv2.boundingRect(cnt)
   if (h > 10 and w > 40) and not(w >= 512 - 5 and h >= 512 - 5):
     cv2.rectangle(Im_Clone2, (x, y), (x+w,y+h), (0,0,255), 1)
     crop = Im_Clone1[y:y+h, x:x+w]
     filename = "%d" %i
#     cv2.imshow(filename, crop)
     i+=1



#cv2.imshow('Mean Adaptive Threshold', Im_Th1)
#cv2.imshow('Gaussian Adaptive Threshold', Im_Th2)
#cv2.imshow('Gradient', Im_Grad)
#cv2.imshow('Close', closing)
#cv2.imshow('Contour 1', Im_Clone1)
#cv2.imshow('Contour 2', Im_Clone2)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

import os
# Directory with our text region pictures
train_text_dir = os.path.join('E:/text region detectoin data set/text region')

# Directory with our non text region pictures
train_non_text_dir = os.path.join('E:/text region detectoin data set/non text region')

train_text_names = os.listdir(train_text_dir)
print(train_text_names[:10])

train_non_text_names = os.listdir(train_non_text_dir)
print(train_non_text_names[:10])

print('total training text images:', len(os.listdir(train_text_dir)))
print('total training non text images:', len(os.listdir(train_non_text_dir)))

import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
   
model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'E:/text region detectoin data set/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=146,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=20,  
      epochs=20)