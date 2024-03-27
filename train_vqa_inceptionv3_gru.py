# Import necessary libraries
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Dropout, concatenate, GRU
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import pickle

# Load data from files
train_file_questions = 'Data/v2_OpenEnded_mscoco_train2014_questions.json'
train_file_annotations = 'Data/v2_mscoco_train2014_annotations.json'
val_file_questions = 'Data/v2_OpenEnded_mscoco_val2014_questions.json'
val_file_annotations = 'Data/v2_mscoco_val2014_annotations.json'

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

# Load image features
with open('train_image_feature_inceptionv3.pkl', 'rb') as fp:
    train_imgs_features = pickle.load(fp)
    fp.close()

with open('val_image_feature_inceptionv3.pkl', 'rb') as fp:
    val_imgs_features = pickle.load(fp)
    fp.close()

train_imgs_features.update(val_imgs_features)

# Combine questions and annotations
train_questions += val_questions
train_annotations += val_annotations

# Extract questions, answers, and image IDs
questions = []
answers = []
image_ids = []

for i in range(len(train_questions)):
    questions.append(train_questions[i]['question'])
    answers.append(train_annotations[i]['multiple_choice_answer'])
    image_ids.append(train_questions[i]['image_id'])

# Tokenize answers and map them to integer labels
tokenizer = Tokenizer()
tokenizer.fit_on_texts(answers)
answer_sequences = tokenizer.texts_to_sequences(answers)
num_classes = len(tokenizer.word_index) + 1  # Add 1 for padding token

# Convert questions to sequences and pad them
max_question_length = 30
question_tokenizer = Tokenizer()
question_tokenizer.fit_on_texts(questions)
question_sequences = question_tokenizer.texts_to_sequences(questions)
padded_sequences = pad_sequences(question_sequences, maxlen=max_question_length)

# Define the model
question_input = Input(shape=(max_question_length,), name='question_input')
question_embedding = Embedding(input_dim=len(question_tokenizer.word_index) + 1, output_dim=300, input_length=max_question_length, name='question_embedding')(question_input)
question_lstm = GRU(units=256, name='question_lstm')(question_embedding)
question_dropout = Dropout(0.2, name='question_dropout')(question_lstm)

output = Dense(units=num_classes, activation='softmax', name='output')(question_dropout)

model = Model(inputs=question_input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Custom data generator
def data_generator(padded_questions, labels, batch_size):
    num_samples = len(labels)
    steps_per_epoch = num_samples // batch_size
    while True:
        for i in range(steps_per_epoch):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_padded_questions = padded_questions[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            yield np.asarray(batch_padded_questions), np.asarray(batch_labels)

# Convert answer sequences to one-hot encoded vectors
one_hot_answers = np.zeros((len(answer_sequences), num_classes))
for i, seq in enumerate(answer_sequences):
    one_hot_answers[i, seq] = 1

# Convert validation questions to sequences and pad them
val_question_sequences = question_tokenizer.texts_to_sequences(val_questions)
val_padded_sequences = pad_sequences(val_question_sequences, maxlen=max_question_length)

# Convert validation answer sequences to one-hot encoded vectors
val_answer_sequences = tokenizer.texts_to_sequences([val_annotations[i]['multiple_choice_answer'] for i in range(len(val_annotations))])
val_one_hot_answers = np.zeros((len(val_answer_sequences), num_classes))
for i, seq in enumerate(val_answer_sequences):
    val_one_hot_answers[i, seq] = 1

# Training
print("Starting model training")
batch_size = 16
steps_per_epoch = len(one_hot_answers) // batch_size
model.fit(data_generator(padded_sequences, one_hot_answers, batch_size),
          steps_per_epoch=steps_per_epoch,
          epochs=300,
          validation_data=data_generator(val_padded_sequences, val_one_hot_answers, batch_size),
          validation_steps=len(val_questions) // batch_size)

# Save the model
model.save("inceptionv3_GRU_Nadam_optimizer.h5")
print("Model saved.")
