'''Authors: 
Debaleen Das Spandan
S.M. Faiaz Mursalin
'''

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from tqdm import tqdm
import re
import pickle
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# define paths to dataset and output paths
TRAIN = 0
VALIDATION = 1

if len(sys.argv) > 1:
    DATA_FOR = TRAIN if sys.argv[1].strip().lower() == "train" else VALIDATION

# DATA_FOR = TRAIN
# DATA_FOR = VALIDATION

MODEL_USED = 'vgg19'

print(
    f"==================== VGG 19 Feature Extraction for {'train images' if DATA_FOR == TRAIN else 'Validation images'} ====================")

image_dir = "Data/train_images" if DATA_FOR == TRAIN else "Data/val_images"
output_file = f"Output/train_features_{MODEL_USED}.npy" if DATA_FOR == TRAIN else f"Output/val_features_{MODEL_USED}.npy"
pickle_file = f'./Pickle files/train_image_feature_{MODEL_USED}.pkl' if DATA_FOR == TRAIN else f"./Pickle files/val_image_feature_{MODEL_USED}.pkl"


# define data generator to preprocess the images
target_size = (224, 224)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    image_dir, target_size=target_size, batch_size=32, class_mode=None, shuffle=False
)

# create a vgg19 model to extract image features
base_model = VGG19(weights="imagenet")
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer("flatten").output)

# extract image features for each image in the training set
train_features = []
for i in tqdm(range(len(generator))):
    batch = generator.next()
    features = model.predict_on_batch(batch)
    train_features.append(features)

if train_features:
    # concatenate and reshape the extracted features into a numpy array
    train_features = np.concatenate(train_features)
    train_features = train_features.reshape((len(generator.filenames), -1))

    # saving the expected feature to numpy file
    np.save(output_file, train_features)

    # save features with IDs in a dictionary in a pkl file and add IDs to features
    img_ids = np.array([int(re.search("[0-9][0-9][0-9][0-9][0-9]+", gen).group())
                       for gen in generator.filenames])
    image_features = {}
    for i in range(len(img_ids)):
        image_features[img_ids[i]] = train_features[i]

    # save dictionary to train_image_feature_inception.pkl file
    with open(pickle_file, 'wb') as fp:
        pickle.dump(image_features, fp)
        print(f'dictionary saved successfully to file: {pickle_file}')
else:
    print('No images found in the specified directory.')
