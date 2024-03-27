# %%writefile custom_model_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.utils import pad_sequences
from keras.applications.inception_v3 import preprocess_input
import pickle
import requests
from PIL import Image
import cv2

# Title
# st.title("Visual Question Answering Demo")


# load model, set cache to prevent reloading
# @st.cache(allow_output_mutation=True)
def load_model():
    # model_incep = tf.keras.models.load_model(r'/content/drive/MyDrive/FInal Project NTI/Models/inception_v3_VQA.h5')
    model_gru = tf.keras.models.load_model(r'./inceptionv3_GRU_Nadam_optimizer.h5')
    # loading
    with open(r'/content/drive/MyDrive/FInal Project NTI/Tokenizers/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(r'/content/drive/MyDrive/FInal Project NTI/Tokenizers/answers_map.pkl', 'rb') as handle:
        answer_tokenizer = pickle.load(handle)
    with open(r'/content/drive/MyDrive/FInal Project NTI/Tokenizers/answers_map.pkl', 'rb') as fp:
        answr_map = pickle.load(fp)

    return model_incep, model_gru, tokenizer, answer_tokenizer, answr_map


with st.spinner("Loading Model...."):
    model_incep, model_gru, tokenizer, answer_tokenizer, answr_map = load_model()
    max_question_length = 30


# image preprocessing
def load_image(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img = tf.image.resize(img, (299, 299))
    img = tf.expand_dims(img, axis=0)
    return img


# Get image URL from user
image_path = st.text_input("Enter Image URL ...",
                           "https://media.istockphoto.com/photos/passenger-airplane-flying-above-clouds-during-sunset-picture-id155439315?k=20&m=155439315&s=612x612&w=0&h=BvXCpRLaP5h1NnvyYI_2iRtSM0Xsz2jQhAmZ7nA7abA=")
question = st.text_input("Enter the question to answer...", "is there a human in the picture?")

if st.button('Refresh'):
    st.experimental_rerun()
# Get image from URL and predict
if image_path:
    try:
        content = requests.get(image_path).content
        st.write("Predicting Answer...")
        with st.spinner("Answering..."):
            resized_img = load_image(content)
            img_features = model_incep.predict(resized_img)
            question_data = tokenizer.texts_to_sequences(question)
            padded_sequences = pad_sequences(question_data, maxlen=max_question_length)
            # -- Predict the answers
            pred_ind = model_gru.predict([np.asarray([padded_sequences[0]]), np.asarray([img_features[0]])])
            st.write("Predicted Answer: ", list(answr_map.keys())[np.argmax(pred_ind)])
            st.image(content, use_column_width=True)
    except:
        st.write("Invalid URL")


# image preprocessing
def load_image1(image):
    img1 = preprocess_input(image)
    resized_img = cv2.resize(img1, (299, 299))
    img2 = np.expand_dims(resized_img, axis=0)
    return img2


# Get uploaded image from user
uploaded_img = st.file_uploader("upload an Image..", type=['png', 'jpg', 'jpeg'])

if uploaded_img is not None:
    img = np.asarray(Image.open(uploaded_img))
    st.image(img)
    try:
        st.write("Predicting Answer...")
        with st.spinner("Answering..."):
            resized_im = load_image1(img)
            img_features = model_incep.predict(resized_im)
            question_data = tokenizer.texts_to_sequences(question)
            padded_sequences = pad_sequences(question_data, maxlen=max_question_length)
            # -- Predict the answers
            pred_ind = model_gru.predict([np.asarray([padded_sequences[0]]), np.asarray([img_features[0]])])
            st.write("Predicted Answer: ", list(answr_map.keys())[np.argmax(pred_ind)])

    except:
        st.write("Invalid URL")