'''Authors: 
Debaleen Das Spandan
S.M. Faiaz Mursalin
'''

# for image feature extraction and training models at once
# run directory setup before running this file

import subprocess


# format:  (file_name_for_feature_extraction, [args: train feature or validation feature], (lstm model, gru model with specific feature extraction method))
file_pairs = [
    ("2_Image_feature_extraction_vgg19.py", [
     "train", "valid"], ("3_train_vqa_vgg19_lstm.py", "3_train_vqa_vgg19_gru.py")),
    ("2_Image_feature_extraction_autoencoder.py", [
     "train", "valid"], ("3_train_vqa_autoencoder_lstm.py", "3_train_vqa_autoencoder_gru.py")),
    ("2_Image_feature_extraction_inceptionv3.py", [
     "train", "valid"], ("3_train_vqa_inceptionv3_lstm.py", "3_train_vqa_inceptionv3_gru.py")),
]

# Function to run a Python file with command line arguments and check its exit status


def run_script(filename, args):
    command = ["python", filename] + args
    print(f"Running {command}..")
    try:
        result = subprocess.run(command, check=True)
        return result.returncode == 0  # Return True if script exits successfully
    except subprocess.CalledProcessError:
        return False


# Iterate over the file pairs
for feature_extraction_file, args, training_files in file_pairs:
    # assume successful run for feature extraction
    outs = True
    for arg in args:
        print(f"Running {feature_extraction_file} with argument {arg}...")
        # if run_script return false, outs will be false
        if not run_script(feature_extraction_file, [arg]):
            print(f"Failed run for {feature_extraction_file} with arg: {arg}")
            outs = False
        # continue running other args for feature extraction, so no break

    # if all args feature extraction was successful:
    # run model training
    if outs:
        for training_file in training_files:
            # run training files
            print(f"Now running {training_file}...")
            if run_script(training_file, []):
                print(f"Successfully ran {training_file}.")
            else:
                print(f"Failed to run {training_file}.")
