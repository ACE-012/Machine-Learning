# %%
import pandas
import numpy as np
import shutil
import matplotlib.pyplot as plt
import math
import os
import glob

# %%
ROOT_DIR="./Datasets/lung"
number_of_images={}
for dir in os.listdir(ROOT_DIR):
    number_of_images[dir]=len(os.listdir(os.path.join(ROOT_DIR,dir)))

# %%
print(number_of_images.items(),sep="\n")
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
os.system('echo $XLA_FLAGS')
os.system('echo $TF_GPU_ALLOCATOR')

# %%
def dataFolder(path,split):
    if not os.path.exists("./"+path):
        os.mkdir("./"+path)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs("./"+path+"/"+dir)
            for img in np.random.choice(a= os.listdir(os.path.join(ROOT_DIR,dir)),
                                    size=(math.floor(split*number_of_images[dir])),
                                    replace=False):
                O=os.path.join(ROOT_DIR,dir,img)
                D=os.path.join("./"+path+"/",dir)
                shutil.copy(O,D)
    else:
        print(path+" folder exists")

# %%
dataFolder("train",0.7)
dataFolder("val",0.5)
dataFolder("test",1)

# %%
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras

# %%
model =Sequential()
model.add(Conv2D(filters=8,kernel_size=(3,3),activation='relu',input_shape=(512,512,3),padding='same'))
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=1,activation='relu'))
model.summary()

# %%
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

# %%
train_data=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,
                   rescale=1/255,horizontal_flip=True).flow_from_directory(directory='./train',target_size=(512,512),batch_size=32,class_mode='binary')

# %%
test_data=ImageDataGenerator(rescale=1/255).flow_from_directory(
    directory='./test',target_size=(512,512),batch_size=32,class_mode='binary')
val_data=ImageDataGenerator(rescale=1/255).flow_from_directory(
    directory='./val',target_size=(512,512),batch_size=32,class_mode='binary')

# %%
from keras.callbacks import ModelCheckpoint,EarlyStopping
es=EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=3,verbose=1,mode='auto')
mc=ModelCheckpoint(monitor="val_accuracy",filepath="./bestmodel.h5",verbose=1,save_best_only=True,mode='auto')
cd=[es,mc]

# %%
hs=model.fit_generator(generator=train_data,
                       steps_per_epoch=16,
                       epochs=10,verbose=1,
                       validation_data=val_data,validation_steps=16,
                       callbacks=cd)


