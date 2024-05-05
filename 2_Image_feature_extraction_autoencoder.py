'''Authors: 
Debaleen Das Spandan
S.M. Faiaz Mursalin
'''

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.applications.inception_v3 import preprocess_input
import re
import pickle
import tqdm
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

TRAIN = 0
VALIDATION = 1

if len(sys.argv) > 1:
    DATA_FOR = TRAIN if sys.argv[1].strip().lower() == "train" else VALIDATION

MODEL_USED = "autoencoder"

print(
    f"==================== Autoencoder Feature Extraction for {'train images' if DATA_FOR == TRAIN else 'Validation images'} ====================")

image_dir = "Data/train_images" if DATA_FOR == TRAIN else "Data/val_images"
output_file = f"Output/train_features_{MODEL_USED}.npy" if DATA_FOR == TRAIN else f"Output/val_features_{MODEL_USED}.npy"
pickle_file = f'./Pickle files/train_image_feature_{MODEL_USED}.pkl' if DATA_FOR == TRAIN else f"./Pickle files/val_image_feature_{MODEL_USED}.pkl"


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


# Example usage
encoder = create_encoder()
print(encoder.summary())


datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    image_dir, target_size=(299, 299), batch_size=32, class_mode=None, shuffle=False
)

# Extract features
features = []
for i in tqdm.tqdm(range(len(generator))):
    batch = generator.next()
    batch_features = encoder.predict_on_batch(batch)
    features.append(batch_features)

if features:
    # Concatenate and reshape the extracted features into a numpy array
    features = np.concatenate(features)
    features = features.reshape((len(generator.filenames), -1))

    # Save the extracted features to a numpy file
    np.save(output_file, features)

    # save features with IDs in a dictionary in a pkl file and add IDs to features
    img_ids = np.array([int(re.search("[0-9][0-9][0-9][0-9][0-9]+", gen).group())
                       for gen in generator.filenames])
    image_features = {}
    for i in range(len(img_ids)):
        image_features[img_ids[i]] = features[i]

    # save dictionary to train_image_feature_inception.pkl file
    with open(pickle_file, 'wb') as fp:
        pickle.dump(image_features, fp)
        print(f'dictionary saved successfully to file: {pickle_file}')

else:
    print('No images found in the specified directory.')
