import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

default_image_size = tuple((128, 128))
model = tf.keras.models.load_model('CNN_model.keras')
#for predicting single image
#image = '/home/trillian/plantvillage/PlantVillage/Tomato_healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG'

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def predict(img_loc):
    image = convert_image_to_array(img_loc)
    image_ = np.array([image], dtype=np.float16) / 225.0
    result = model.predict(image_)[0][0]
    if result>0.5:
        return 1
    else:
        return 0

#predict(image)
