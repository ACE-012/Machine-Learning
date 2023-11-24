from typing import Any
import numpy as np
import tensorflow as tf
from custom_fn import specificity,recall
import pathlib
class_names=['Malignant', 'Normal   ']
custom_objects = {"specificity": specificity, "recall": recall}

model:Any=tf.keras.models.load_model('./bestmodel90_10.keras',custom_objects=custom_objects)


data_dir = pathlib.Path('./Datasets/lung/').with_suffix('')
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.5,
  subset="validation",
  label_mode="binary",
  seed=123,
  image_size=(512, 512),
  batch_size=32)


from keras.preprocessing.image import load_img,img_to_array
path="./Datasets/lung/Malignant cases/Malignant case (554).jpg"
img=load_img(path,target_size=(512,512))

arr=img_to_array(img)
arr=np.expand_dims(arr,axis=0)
pred=model.predict(arr)[0][0]
print(model.evaluate(val_ds))
print(f'{class_names[0]}\t:{(1-pred)*100:.2f}%')
print(f'{class_names[1]}\t:{(pred)*100:.2f}%')