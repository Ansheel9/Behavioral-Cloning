import numpy as np
import csv
import cv2
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, GlobalAveragePooling2D
from keras.applications import xception
from keras import backend as K
from keras.utils import np_utils

### Reading the csv file containing the patch of the images, steering and few other parameters ###
lines = []
with open('data/driving_log.csv') as csvfile:
    Reader = csv.reader(csvfile)
    for line in Reader:
        lines.append(line)

### Storing the images and their steering measurements into arrays ###
images = []
measurements = []
correction_factor = 0.2
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2. imread(current_path)
        images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction_factor)
    measurements.append(measurement-correction_factor)       
        
### Augmenting the recorded data with preprocessed data ###
aug_images = []
aug_measurements = []

for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = float(measurement) * (-1.0)
    aug_images.append(flipped_image)
    aug_measurements.append(flipped_measurement)

### Normalizing & Cropping the record ###
final_images = []
final_measurements = []

for image, measurement in zip(aug_images, aug_measurements):
    normalized_image = (image / 127.5) - 1.
    cropped_image = normalized_image[65:140,:]
    final_images.append(cropped_image)
    final_measurements.append(measurement)

X_train = np.array(final_images)
y_train = np.array(final_measurements)

### Model Architecture ###

model = xception.Xception(weights='imagenet', include_top=False, input_shape=(75, 320, 3))
print (model.summary())

### New Layers which will be trained on our data set and will be stacked over the Xception Model ###
x=model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
output=Dense(1,activation='linear')(x)

New_Model=Model(model.input,output)

### Freezing all the Imported Layers ###
for layers in model.layers:
	layers.trainable=False

New_Model.compile(loss = 'mse', optimizer = 'adam')
New_Model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 6)

### Saving the model ###
New_Model.save('model.h5')
print('Model Saved')