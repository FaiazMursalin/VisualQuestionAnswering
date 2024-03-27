import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from tqdm import tqdm
import re
import pickle

# define paths to dataset and output paths
image_dir = "Data/val_images"
output_file = "Output/val_features_vgg19.npy"

# define data generator to preprocess the images
target_size = (224, 224)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    image_dir, target_size=target_size, batch_size=32, class_mode=None, shuffle=False
)

# create a vgg19 model to extract image features
base_model = VGG19(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("flatten").output)

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
    img_ids = np.array([int(re.search("[0-9][0-9][0-9][0-9][0-9]+", gen).group()) for gen in generator.filenames])
    image_features = {}
    for i in range(len(img_ids)):
        image_features[img_ids[i]] = train_features[i]

    # save dictionary to train_image_feature_vgg19.pkl file
    with open('val_image_feature_vgg19.pkl', 'wb') as fp:
        pickle.dump(image_features, fp)
        print('dictionary saved successfully to file')
else:
    print('No images found in the specified directory.')
