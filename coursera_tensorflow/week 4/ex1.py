import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255) #rescale to normalize the data

train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size = (300, 300),
      batch_size = 128,
      class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255) #rescale to normalize the data

validation_generator = test_datagen.flow_from_directory(
      validation_dir,
      target_size = (300, 300),
      batch_size = 128,
      class_mode = 'binary')   


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                         input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
  
model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr = 0.001), #learning rate
              metrics = ['acc']
              )

history = model.fit_generator(
      train_generator,
      step_per_epoch = 8, #batch size 128, so in order to load 1024 images need to do 8 batches, 128X8=1024
      epochs = 15,
      validation_data = validation_generator,
      validation_steps = 8, #256 images and want to handle them in batches of 32, so need 8 steps, 32X8=256
      verbose = 2)