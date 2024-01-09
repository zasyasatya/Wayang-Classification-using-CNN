import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

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

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (200, 200, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),                                
                                    #
                                    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),                                
                                    #                                     
                                    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),                                    
                                    #                                    
                                    tf.keras.layers.Flatten(),                                    
                                    # 
                                    tf.keras.layers.Dense(512, activation = 'relu'),                                    
                                    #                                    
                                    tf.keras.layers.Dense(66, activation = 'softmax')
                                    
    ])



# Creating a neural convolutional layer
# model = Sequential()

# # 1 convolutional layer 0 1 2 3 (60000, 28, 28, 1) We skipped the [0] layer -> 60000 ; --> 28 - 3 + 1= 26 x 26
# model.add(Conv2D(64, (3, 3), input_shape = trainData.shape[1:])) #only for first convolution layer to mention input layer size (64 diff kernel, )
# model.add(Activation('relu')) #activation fuction --> if < 0 then value.remove() else if > 0 then value.allowd
# model.add(MaxPooling2D(pool_size = (2, 2))) #max pooling single maximum value of 2 x 2


# # 2 convolution layer --> 26 - 3 + 1 = 24 x 24
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# # 3 convolution layer --> 
# model.add(Conv2D(64, (3, 3))) #--> 24 x 24
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# # Fully connected layer #1 --> 20 x 20 = 400
# model.add(Flatten()) # before using fully connected layer, need to be flatten
# model.add(Dense(64)) # we have 64 neurons
# model.add(Activation('relu'))

# # Fully connected layer # 2
# model.add(Dense(32))
# model.add(Activation('relu')) 


# If we use binary classification (1 output)
# Last fully connected layer, output must be equal to number of classes, 10 (0-9)
# model.add(Dense(10)) # equal to 10 classes
# model.add(Activation('softmax')) # activation function for class probabilities



# model_fit = model.fit(trainData,
#                       # batch_size=128,
#                       steps_per_epoch = 3,
#                       epochs = 200,
#                       validation_data = testData)


# model.save_weights('testing1')

model.load_weights('D:/KULIAH/SEMESTER 6/PCD Lanjut/CNN/testing1')
# print(model.get_weights())

# loss, acc = model.evaluate(trainData)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# finalScore = model.evaluate(trainData, testData, verbose = 0)
# print("%s: %.2f%%" % (model.metrics_names[1], finalScore[1]*100))

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.001), metrics = ['accuracy'])
loss, acc = model.evaluate(trainData, testData, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

imageForPredict = image.load_img("D:/dataset/wayang-arjuna.jpg", target_size = (200, 200))
plt.imshow(imageForPredict)
plt.show()

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
