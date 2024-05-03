import tensorflow as tf
import numpy as np
from keras.utils import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocessing
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg_preprocessing
import pickle
import cv2

IMG_FEATURE_MODEL = 'autoencoder'
TEXT_MODEL_USED = 'gru'
TRAINING_EPCOHS = 200

image_path = "./Data/test2015/COCO_test2015_000000000063.jpg"
question = "What is the color of field?"

def create_encoder():
    # Input layer for the images (adapt size to your needs)
    inputs = Input(shape=(299, 299, 3))
    # Example Convolutional Layers (you can customize this part as needed)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Flatten and dense layers to produce the final feature vector
    x = Flatten()(x)
    encoded = tf.keras.layers.Dense(512, activation='relu')(x)
 
    # Create model
    encoder = Model(inputs, encoded, name='encoder')
    return encoder

def load_feature_extractor(which):
    if which == "inceptionv3":
        base_model = InceptionV3(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        return model, inception_preprocessing, (299, 299)
    elif which == "autoencoder":
        return create_encoder(), inception_preprocessing, (299, 299)
    elif which == "vg19":
        base_model = VGG19(weights="imagenet")
        model = Model(inputs=base_model.input, outputs=base_model.get_layer("flatten").output)
        return model, vgg_preprocessing, (224, 224)
    else:
        raise SystemExit(f"{which} feature extractor not supported.")


def extract_image_feature(img_path, which_extractor):
    img = cv2.imread(img_path)
    img = tf.cast(img, tf.float32)
    feature_extractor, preprocess_input, target_shape = load_feature_extractor(which_extractor)
    img = preprocess_input(img)
    img = tf.image.resize(img, target_shape)
    img = tf.expand_dims(img, axis=0)
    return feature_extractor.predict(img)



model = tf.keras.models.load_model(f"./Keras models/{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}_nadam_optimizer-run{TRAINING_EPCOHS}epochs.keras")
max_question_length = 30

with open(f"./Pickle files/tokenizer_{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}.pkl", "rb") as infile:
    tokenizer = pickle.load(infile)

with open(f"./Pickle files/label_map_{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}.pkl", "rb") as infile:
    answer_map = pickle.load(infile)


img_features = extract_image_feature(image_path, IMG_FEATURE_MODEL)

# tokenizer.fit_on_texts(answers)
question_data = tokenizer.texts_to_sequences(question)
padded_sequences = pad_sequences(question_data, maxlen=max_question_length)
print(question_data)
# -- Predict the answers
pred_ind = model.predict([np.asarray([padded_sequences[0]]), np.asarray([img_features[0]])])
print("Predicted Answer: ", list(answer_map.keys())[np.argmax(pred_ind)])

