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
    # Note the input shape is the desired size of the image 224x224 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(filters = 96, kernel_size = (3, 3), strides = (4, 4), padding = 'same', activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),

    # The second convolution
    tf.keras.layers.Conv2D(filters = 256, kernel_size = (11, 11), strides = (1, 1), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),

    # The third convolution
    tf.keras.layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),

    # The fourth convolution
    tf.keras.layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),

    # The fifth convolution
    tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'),
    tf.keras.layers.BatchNormalization(),

    # Flatten the results to feed into a Dense Layer
    tf.keras.layers.Flatten(),
    # The first 4096 Dense layer
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),

    # The second 4096 Dense layer
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),

    # The second 1000 Dense layer
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),

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
        target_size=(224, 224),  # All images will be resized to 150x150
        batch_size=146,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=20,
      verbose=1,
      epochs=20)