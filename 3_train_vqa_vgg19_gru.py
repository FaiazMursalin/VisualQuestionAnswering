'''Authors: 
Debaleen Das Spandan
S.M. Faiaz Mursalin
'''


import json
import pickle
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, savefilename):
    plt.figure(figsize=(12, 6), dpi=300.0)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc="upper left")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc="upper left")

    plt.tight_layout()
    plt.grid()
    plt.savefig(savefilename)


NUM_EPOCHS = 200

IMG_FEATURE_MODEL = 'vgg19'
TEXT_MODEL_USED = 'gru'

train_file_questions = './Data/v2_OpenEnded_mscoco_train2014_questions.json'
train_file_annotations = './Data/v2_mscoco_train2014_annotations.json'
val_file_questions = './Data/v2_OpenEnded_mscoco_val2014_questions.json'
val_file_annotations = './Data/v2_mscoco_val2014_annotations.json'
test_file_questions = './Data/v2_OpenEnded_mscoco_test2015_questions.json'

with open(train_file_questions, 'r') as f:
    train_questions = json.load(f)['questions']
    f.close()

with open(train_file_annotations, 'r') as f:
    train_annotations = json.load(f)['annotations']
    f.close()

with open(val_file_questions, 'r') as f:
    val_questions = json.load(f)['questions']
    f.close()

with open(val_file_annotations, 'r') as f:
    val_annotations = json.load(f)['annotations']
    f.close()

# Read train dictionary pkl file
# change file location
with open(f'./Pickle files/train_image_feature_{IMG_FEATURE_MODEL}.pkl', 'rb') as fp:
    train_imgs_features = pickle.load(fp)
    print('successful')
# Read validation dictionary pkl file
# change file location
with open(f'./Pickle files/val_image_feature_{IMG_FEATURE_MODEL}.pkl', 'rb') as fp:
    val_imgs_features = pickle.load(fp)
    print('successful')

# append validate to train features
train_imgs_features.update(val_imgs_features)
print('length of train image features, ', len(train_imgs_features))

# append validate question and answers to train questions and answer
# Combine the training and validation questions and annotations
train_questions += val_questions
train_annotations += val_annotations

# ENCODE QUESTIONS AND ANSWER AND CREATE IMAGE FEATURES LIST
# extract the questions and answers
questions = []
answers = []
features_id = []

for i in range(len(train_questions)):
    questions.append(train_questions[i]['question'])
    answers.append(train_annotations[i]['multiple_choice_answer'])
    features_id.append(train_questions[i]['image_id'])

# TOKENIZATION
tokenizer = Tokenizer()
tokenizer.fit_on_texts(answers)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(questions)

max_question_length = 30
padded_sequences = pad_sequences(sequences, maxlen=max_question_length)

# Indexing
answers_tokenizer = Tokenizer()
answers_tokenizer.fit_on_texts(answers)
answer_word_index = answers_tokenizer.word_index
num_of_classes = len(answer_word_index)
answer_sequences = answers_tokenizer.texts_to_sequences(answers)
with open(f"./Pickle files/answer_sequences_{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}.pkl", "wb") as outfile:
    pickle.dump(answer_sequences, outfile)

# pad the answers sequences so that they all have the same length
max_answer_length = max(len(seq) for seq in answer_sequences)
padded_answers = pad_sequences(answer_sequences, maxlen=max_answer_length)

# Get the unique answers in the dataset and create a dictionary to map them to integer labels
unique_answers = list(set(answers))
print("len of unique answers: ", len(unique_answers))
label_map = {answer: i for i, answer in enumerate(unique_answers)}
with open(f"./Pickle files/label_map_{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}.pkl", "wb") as outfile:
    pickle.dump(label_map, outfile)

with open(f"./Pickle files/tokenizer_{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}.pkl", "wb") as outfile:
    pickle.dump(tokenizer, outfile)

# def batch_labels(labels)

# Convert the answers to integer labels and then to one hot vector
labels = [label_map[answer] for answer in answers]

print(labels[:10])

print("len of labels: ", len(labels))
# one_hot_answers = tf.keras.utils.to_categorical(
# labels, num_classes=len(unique_answers))

print('features_id: ', len(features_id))
print('shape of padded sequence: ', padded_sequences.shape)
# print('one hot answer shape: ', one_hot_answers.shape)


# split inplace 70-30 train test
split_indices = np.random.randint(low=0, high=len(
    features_id), size=int(len(features_id) * 0.3))
split_indices = sorted(split_indices, reverse=True)

test_padded_sequences = []
padded_sequences = list(padded_sequences)
for i in split_indices:
    test_padded_sequences.append(padded_sequences.pop(i))

# test_one_hot_answers = []
# one_hot_answers = list(one_hot_answers)
# for i in split_indices:
# test_one_hot_answers.append(one_hot_answers.pop(i))

test_labels = []
for i in split_indices:
    test_labels.append(labels.pop(i))

test_features_id = []
for i in split_indices:
    test_features_id.append(features_id.pop(i))

# split 30% test into 20%test and 10% validation in place
split_indices = np.random.randint(low=0, high=len(
    test_features_id), size=int(len(test_features_id) * 0.3))
split_indices = sorted(split_indices, reverse=True)

val_padded_sequences = []
for i in split_indices:
    val_padded_sequences.append(test_padded_sequences.pop(i))

val_labels = []
for i in split_indices:
    val_labels.append(test_labels.pop(i))

val_features_id = []
for i in split_indices:
    val_features_id.append(test_features_id.pop(i))

# Define the input layers
question_input = Input(shape=(max_question_length,), name='question_input')
image_input = Input(shape=(25088,), name='image_input')

# Define the embedding layer for the questions
question_embedding = Embedding(input_dim=len(word_index) + 1, output_dim=300, input_length=max_question_length,
                               name='question_embedding')(question_input)

# Define the GRU layer for the questions
question_gru = tf.keras.layers.GRU(
    units=256, name='question_gru', return_sequences=True)(question_embedding)
question_gru = Dropout(0.2, name='question_dropout')(question_gru)
question_gru2 = tf.keras.layers.GRU(
    units=256, name='question_gru2')(question_gru)
question_gru2 = Dropout(0.2, name='question_dropout2')(question_gru2)

# Define the dense layer for the image features
image_dense = Dense(units=256, activation='relu',
                    name='image_dense')(image_input)
image_dense = Dropout(0.2, name='image_dropout')(image_dense)

# Concatenate the output from the LSTM and dense layers
dense_1 = concatenate([question_gru2, image_dense], name='concatenated')
dense_2 = Dense(512, activation='relu')(dense_1)
dense_3 = Dense(256, activation='relu')(dense_2)
# Define the output layer for the classification

output = Dense(units=len(unique_answers),
               activation='softmax', name='output')(dense_3)

# Define the model
model = Model(inputs=[question_input, image_input], outputs=output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='nadam', metrics=['accuracy'])


def data_generator(image_features, padded_questions, labels, batch_size):
    num_samples = len(labels)
    steps_per_epoch = num_samples // batch_size
    while True:
        for i in range(steps_per_epoch):
            batch_image_features = []
            for j in image_features[i * batch_size:(i + 1) * batch_size]:
                batch_image_features.append(train_imgs_features[j])
            batch_padded_questions = padded_questions[i *
                                                      batch_size:(i + 1) * batch_size]
            batch_labels = labels[i * batch_size:(i + 1) * batch_size]
            yield [np.asarray(batch_padded_questions), np.asarray(batch_image_features)], np.asarray(batch_labels)


print("starting model training")
batch_size = 32  # 128
steps_per_epoch = len(labels) // batch_size

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f"Checkpoints/{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}_{NUM_EPOCHS}", save_best_only=True)

history = model.fit(data_generator(features_id, padded_sequences, labels, batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=NUM_EPOCHS,
                    validation_data=data_generator(
                        val_features_id, val_padded_sequences, val_labels, batch_size),
                    validation_steps=int(len(val_features_id) / batch_size),
                    callbacks=[checkpoint])


model.save(
    f"./Keras models/{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}_nadam_optimizer-run{NUM_EPOCHS}epochs.keras")
model.save(
    f"./H5 models/{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}_nadam_optimizer-run{NUM_EPOCHS}epochs.h5")
print("model saved")

plot_history(
    history, f"./PNG files/{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}_nadam_optimizer-run{NUM_EPOCHS}epochs.png")

with open(f"./Pickle files/{IMG_FEATURE_MODEL}_{TEXT_MODEL_USED}_nadam_optimizer-run{NUM_EPOCHS}epochs.pkl", "wb") as outfile:
    pickle.dump(history, outfile)
