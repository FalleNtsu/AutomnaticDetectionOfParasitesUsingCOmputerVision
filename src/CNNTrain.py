import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

class CNNTrain():
    def __init__(self) -> None:
        pass

    def TrainCNN(self): 
      epochs = 2

      trainFilePath = './AutomnaticDetectionOfParasitesUsingCOmputerVision/Images/Train'
      ValidationFilePath = './AutomnaticDetectionOfParasitesUsingCOmputerVision/Images/Validation'
      cwd = os.getcwd()

      imageW, imageH = 150, 150
      sizeBatch = 32
      samplePerEpoch = 100
      stepsValidation = 100
      filter1 = 32
      filter2 = 64
      consize1 = 3
      consize2 = 2
      poolSize = 2
      classess = 3
      lr = 0.0004

      model = Sequential()
      model.add(Convolution2D(filter1, consize1, consize1,  input_shape=(imageW, imageH, 3)))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

      model.add(Convolution2D(filter2, consize2, consize2))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

      model.add(Flatten())
      model.add(Dense(256))
      model.add(Activation("relu"))
      model.add(Dropout(0.5))
      model.add(Dense(classess, activation='softmax'))

      model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=lr),
                    metrics=['accuracy'])

      trainData = ImageDataGenerator(
          rescale=1. / 255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)

      imageDataGen = ImageDataGenerator(rescale=1. / 255)

      trainGen = trainData.flow_from_directory(
          trainFilePath,
          target_size=(imageH, imageW),
          batch_size=sizeBatch,
          class_mode='categorical')

      validationGen = imageDataGen.flow_from_directory(
          ValidationFilePath,
          target_size=(imageH, imageW),
          batch_size=sizeBatch,
          class_mode='categorical')


      log_dir = './tf-log/'
      tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
      cbks = [tb_cb]

      model.fit(
          trainGen,
          steps_per_epoch=samplePerEpoch,
          epochs=epochs,
          validation_data=validationGen,
          callbacks=cbks,
          validation_steps=stepsValidation)

      target_dir = './models/'
      if not os.path.exists(target_dir):
        os.mkdir(target_dir)
      model.save('./models/model.h5')
      model.save_weights('./models/weights.h5')