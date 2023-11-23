import tensorflow as tf
import pathlib
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
from custom_fn import specificity,recall
# ------------------------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5300)]
    )
# ------------------------------------------------------------------------------------------


data_dir = pathlib.Path('./Datasets/lung/').with_suffix('')

train_ds,val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.3,
  subset="both",
  label_mode="binary",
  seed=123,
  image_size=(512, 512),
  batch_size=32)

model=Sequential()
model.add(Conv2D(filters= 8,kernel_size=(4,4),input_shape=(512,512,3),padding='same'))
model.add(Conv2D(filters= 16,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters= 32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(
  optimizer='adam',
  loss=keras.losses.BinaryCrossentropy(),
  metrics=['accuracy',specificity,recall])

es=EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=6,verbose=1)
mc=ModelCheckpoint(monitor="val_accuracy",filepath="./bestmodel.h5",save_best_only=True)
cd=[es,mc]

model.fit(train_ds,
  validation_data=val_ds,   epochs=30,
  use_multiprocessing=True, callbacks=cd)