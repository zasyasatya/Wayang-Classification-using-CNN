import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import datetime

#normalization
train = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)

#labelingImage
trainData = train.flow_from_directory('D:\dataset\DatasetWayang\A_Data_Train',
                                      target_size = (200, 200),
                                      batch_size = 5, 
                                      # classes = 
                                      class_mode = 'categorical')


testData = test.flow_from_directory('D:\dataset\DatasetWayang\A_Data_Test',
                                      target_size = (200, 200),
                                      batch_size = 5, 
                                      class_mode = 'categorical')

model = tf.keras.models.load_model("wayang.h5")

# date_now = datetime.datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")

# logdir = os.path.join("logs", date_now)
# tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# lokasiModel = "wayang.h5"

# model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.001), metrics = ['accuracy'])

# model_fit = model.fit(trainData,
#                       # batch_size=128,
#                       steps_per_epoch = 3,
#                       epochs = 200,
#                       validation_data = testData,
#                       callbacks = [ ModelCheckpoint(lokasiModel, verbose=1, save_best_only=True), tensorboard_callback])


# model.save_weights('testing1')

# model.load_weights('D:/KULIAH/SEMESTER 6/PCD Lanjut/CNN/testing1')
# # print(model.get_weights())

# loss, acc = model.evaluate(trainData)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# finalScore = model.evaluate(trainData, testData, verbose = 0)
# print("%s: %.2f%%" % (model.metrics_names[1], finalScore[1]*100))

# loss, acc = model.evaluate(trainData, testData, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

imageForPredict = image.load_img("D:/dataset/wayang-arjuna.jpg", target_size = (200, 200))

imageArray = image.img_to_array(imageForPredict)
imageArray = np.expand_dims(imageArray, axis = 0)
# print(imageArray)
imageRes = np.vstack([imageArray])
# print(imageRes)
# val = np.argmax([imageRes])
val = model.predict(imageRes)
# print(val)
valPredict = np.argmax([val])
print("Ini adalah wayang jenis: " + str(list(trainData.class_indices.keys())[list(trainData.class_indices.values()).index(valPredict)]))

plt.imshow(imageForPredict)
plt.show()