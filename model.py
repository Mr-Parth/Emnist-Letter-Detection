import scipy.io as sc
import keras 
import numpy
import matplotlib.pyplot as plt
from keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.utils import to_categorical

from keras import backend as K
K.set_image_dim_ordering('th')


dataset=sc.loadmat("emnist-letters.mat")

x_train=dataset["dataset"][0][0][0][0][0][0].astype("float32")
x_train/=255
y_train=dataset["dataset"][0][0][0][0][0][1].astype("float32")
y_train=y_train.reshape(y_train.shape[0],)
y=to_categorical(y_train)


##Order "A" for order in Fortran form if x if Fortran contiguous in mem
x=x_train.reshape(x_train.shape[0],1,28,28,order="A").astype("float32")

def encode2letter(num):
    return chr(96+num)


plt.imshow(x[2000][0],cmap="gray")

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape=(1,28,28),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 256, activation = 'relu'))
model.add(Dense(output_dim = 27, activation = 'softmax'))

model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x,y,epochs=25,batch_size=200)

x_test=dataset["dataset"][0][0][1][0][0][0].astype("float32")
x_test/=255
y_test=dataset["dataset"][0][0][1][0][0][1].astype("float32")
y_test=y_test.reshape(y_test.shape[0],)
y_=to_categorical(y_test)

x_=x_test.reshape(x_test.shape[0],1,28,28,order="A").astype("float32")

pred=model.predict(x_)
pred=np.argmax(pred,axis=1)
pred=np.float32(pred)


from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,pred)



##Save Model and Weights

json_file=model.to_json()
with open("model.json","w+") as json_model:
    json_model.write(json_file)
    
model.save_weights("model.h5")






