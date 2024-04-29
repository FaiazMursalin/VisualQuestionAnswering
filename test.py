import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import pickle
import requests
from PIL import Image
import cv2

image_path = "./Data/train_images/train2014/COCO_train2014_000000458752.jpg"
question = "What is the color of field?"

def load_image(img_path):
    # img = tf.image.decode_jpeg(image, channels=3)
    img = cv2.imread(img_path)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img = tf.image.resize(img, (299, 299))
    img = tf.expand_dims(img, axis=0)
    return img

model_gru = tf.keras.models.load_model(r'./inceptionv3_GRU_Nadam_optimizer-run200epochs.keras')
max_question_length = 30

with open("tokenizer_incepv3_gru.pkl", "rb") as infile:
    tokenizer = pickle.load(infile)

with open("./label_map_incepv3_gru.pkl", "rb") as infile:
    answer_map = pickle.load(infile)

# print(answer_map)

# raise SystemExit
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


resized_img = load_image(image_path)
img_features = model.predict(resized_img)
# print(img_features)

# tokenizer.fit_on_texts(answers)
question_data = tokenizer.texts_to_sequences(question)
padded_sequences = pad_sequences(question_data, maxlen=max_question_length)
print(question_data)
# -- Predict the answers
pred_ind = model_gru.predict([np.asarray([padded_sequences[0]]), np.asarray([img_features[0]])])
print("Predicted Answer: ", list(answer_map.keys())[np.argmax(pred_ind)])
print("Predicted Answer: ", list(answer_map.values())[np.argmax(pred_ind)])

