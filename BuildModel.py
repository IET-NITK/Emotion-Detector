''' Building CNN model using keras '''

import numpy as np
from sklearn.cross_validation import train_test_split
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
import pandas as pd
from keras import optimizers
from keras.utils import to_categorical

#reading and preparing data
data=pd.read_csv('fer2013.csv', sep=',',header=0)
X=np.array(data['pixels'])
Y=np.array(data['emotion'])
n=X.shape[0]
Xi=[]
for i in range(0,n):
    temp=[]
    for j in X[i].split():
        temp.append(int(i))
    Xi.append(temp)
Xnp=np.array(Xi).reshape(-1,48,48,1)
Y=to_categorical(Y)


#train test split
X_train,X_test,y_train,y_test=train_test_split(Xnp,Y)

 
''' Building model ''' 
model = Sequential()
num_classes=7

#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
 
model.add(Dense(num_classes, activation='softmax'))

# visualise model
model.summary()


#optimiser and loss function
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# training the model
model.fit(X_train,y_train,epochs=2,batch_size=50,validation_data=(X_test,y_test))

#saving the trained model
model.save('my_model.h5')
np.save('pixedarray',Xi)

