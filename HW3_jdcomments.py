
import pandas as pd
path =r'C:/Users/awolpert/OneDrive - Roosevelt University/Desktop/Courses/CST 361-461_408 Deep Learning/DL-2021/Studnets/Programming/Data/housing.txt' #set the path to the datafile
data = pd.read_csv(path , header=None #Lines 4 through 16 are the labels we will use to label all the data coming into and out from the model, each of the labels provided are factors for what is to be expected in a household just that these variables will determine what each house will have and which brackect of income it will be allocated to.
                   , names=[
                        'longitude'
                        ,'latitude'
                        ,'housing_median_age'
                        ,'total_rooms'
                        ,'total_bedrooms'
                        ,'population'
                        ,'households'
                        ,'median_income'
                        ,'median_house_value'
                        ,'ocean_proximity'
                    ]) #input the data into a dataframe


import numpy as np

data = data.dropna(axis=0) #Lines 21 to 24, so this section is a function that will be used to create our array table from the data we will be inputing from the housing.txt file witht the data drop and shape being functions top create the shape of the data here the shape is set to 0 so there will be no changes made that will affect the output of the shape and the collison set to 1 so if collision occurs throuigh our network it
cols = data.shape[1]
r_features = np.array(data.iloc[:,0:cols-2],np.float32)
rows = r_features.shape[0]

from keras.utils import to_categorical #
labels = data['ocean_proximity']
labels_strings, labels_ints = np.unique(labels, return_inverse = True)
categorical_labels = to_categorical(labels_ints, dtype="float32")

from sklearn.preprocessing import StandardScaler #
scale = StandardScaler()
features = scale.fit_transform(r_features)

from numpy.random import default_rng #This function Lines 35 through 40 provides random numbers being produce to simulate real liufe varaibales that can occur that soul;d be taken intop accopunt when going throught he model otherwise the output will be static as there is no randomness into play when going through the training. Here we have it set to 1000 witht he rest fo the function forming an index of data that will be trained and collected through the model.
from numpy import random
np.random.seed(1000)
rand_idx = default_rng().choice(rows, size = rows, replace=False)
features = features[rand_idx]
categorical_labels = categorical_labels[rand_idx] #all the labels

small = random.uniform(0,.5) #This section Lines 42 to 51 this is where we have create our training  model that will be used to train our network fropm the values that we have isnerted that it will go through from the text document. It will be training itself to we set our training size to bve small as we aren't working with alot of data for this homework and we set it to a ranmdomly train with varilabnles we input in. Which case we put .5 and the rest set to zero so there isn't much drastic changes made to our training unit for the network as all we are training is just a small batch of data in the text document for the course of this homework so tgere isn't need to do anything else other the input the data in and let it ttreain itself so that it can then go through actiuvations through ReLu or sigmaid or Softmax before being plotted down into the model witht he losss and accuracy it has collected from the training.
train_size = int((1-small) * rows)
val_size = train_size+int(small/2 * rows)
features_train = features[0:train_size,:]
features_val = features[train_size:val_size,:]
features_test = features[val_size:features.shape[0],:]

categorical_labels_train = categorical_labels[0:train_size]
categorical_labels_val = categorical_labels[train_size:val_size]
categorical_labels_test = categorical_labels[val_size:features.shape[0],:]

print(categorical_labels_train.shape[0] + categorical_labels_val.shape[0] + categorical_labels_test.shape[0])
print(data.shape[0])


from keras import models
from keras import layers

model = models.Sequential() #This sections Lines 60 through 70 is where we add the models and their density and this will form the basis of our model that will take the trainingdone in the previous section to learn from what values we implmented to have for training. In turn this is where we begin to build our model network that will garner how this will be showcased in the end and depedning on the training we will select which activation we are going use to distribute the data in the model which will come into in the next section where we begiun plotting out the data that was training earlier into the model in this case we chose ReLu and softmax for the activation for this homework. We will also be presetned with the accuracy of the training onto the model to better know if we have managed to correct setup the network without any issue of data being greater than required especially for the loss.

model.add(layers.Dense(64, activation='relu', input_shape=(8,)))
model.add(layers.Dense(48, activation='relu'))
model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer='rmsprop'
              ,loss='categorical_crossentropy'
              ,metrics=['accuracy'])
print(model.summary())


size_batch = random.randint(1024-128)+128 #For lines 73 to 84 I'll combine the two sections in one comment. So for the first section lines 73-77 this is where we create our batch size of our model we are going to be using based on the configurations of the earlier section specifically from the training data that we are builkding up from whatr values we put in earlier for this model. In this case we are doing a 128x128 batch size of the data that has been trained previously and gone through the activation layers that wiull be bult upon through our training that we have computed from the values we set up followed by the data from the fiel that will be inpoutted in. The other section Lines 79-84 is where we have setup our labels and functions for our accuracy of the data from our training as well as the loss thnat we will accumluate and this will servie its purpose in the next section of the model as the culmination of the data inputed that has been trained with outs of loss and accuracy will be then plotted onto our model size given the batch size f our data that we have decided trained in our network.
num_epochs = random.randint(100)+10
history=model.fit(features_train,
                    categorical_labels_train, epochs=num_epochs, batch_size = size_batch,
                    validation_data=(features_val, categorical_labels_val))

history_dict = history.history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

import matplotlib.pyplot as plt #Here for lines 86 to 104 this is where we plot the all the data that has been outputted from our netowrk. After the training has been finished and the the accurracy and loss are correect and minimal this is where each point gets plotyted into our model the first half is plotting the Training losses that the mdoel has gone through into our model to showcase how well the network has trained under the values put by us that was used when we inpu the text document into the netowkr.The plots will be based betweeen 0 and 1, if its beyoind that then the training has gone awry and needs to be adjusted. The second half is to plot the accuracy of the data that was output from what the training model could gather and it would be another layer on the model that we have made in an earlier function. The
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()


plt.plot(epochs, acc, 'ro', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.clf()

test_loss, test_acc = model.evaluate(features_test, categorical_labels_test) #Lines 106 to 108 is the test results of the loss that has been factored in threough the training model being printed out to show what loss it has garnered from the current model. It needs to be either between 0 to 1  otherwise if its anything beyoin dand the model is in need of being reevaluated.
print('test_loss:', test_loss)
print('test_acc:', test_acc)
