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

