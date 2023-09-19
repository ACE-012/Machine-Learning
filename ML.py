# %%
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Dense

# %%
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.system('export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda')
os.system('echo $XLA_FLAGS')

# %%
print(tf.__version__)

# %%
data_dir = pathlib.Path('./Datasets/lung_image_sets/').with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpeg')))
print(image_count)

# %%
bengin=list(data_dir.glob('lung_n/*'))
PIL.Image.open(str(bengin[1]))

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(512, 512),
  batch_size=32)

# %%
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(512, 512),
  batch_size=32)

# %%
class_names_ = train_ds.class_names
print(class_names_)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names_[labels[i]])
    plt.axis("off")

# %%
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# %%
num_classes = 3

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(4, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(4, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(4, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
# model=Sequential()
# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(512,512,3)))
# model.add(MaxPool2D(2,2))
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# model.add(Dense(10,activation='softmax'))

# %%
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# %%
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10
)


