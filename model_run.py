from typing import Any
import numpy as np
import tensorflow as tf
from custom_fn import specificity,recall
class_names=['Malignant', 'Normal   ']
custom_objects = {"specificity": specificity, "recall": recall}

model:Any=tf.keras.models.load_model('./bestmodel.h5',custom_objects=custom_objects)

from keras.preprocessing.image import load_img,img_to_array
path="./Datasets/lung/Malignant cases/Malignant case (554).jpg"
img=load_img(path,target_size=(512,512))

arr=img_to_array(img)
arr=np.expand_dims(arr,axis=0)
pred=model.predict(arr)[0][0]
print(f'{class_names[0]}\t:{(1-pred)*100:.2f}%')
print(f'{class_names[1]}\t:{(pred)*100:.2f}%')


