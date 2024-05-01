import subprocess
import sys
 
# Define file pairs and args
file_pairs = [
    ("Image_feature_extraction_vgg19.py", ["train", "valid"], ("train_vqa_vgg19_lstm.py", "train_vqa_vgg19_gru.py")),
    ("image_feature_extraction_autoencoder.py", ["train", "valid"], ("train_vqa_autoencoder_lstm.py", "train_vqa_autoencoder_gru.py")),
    ("Image_feature_extraction_inceptionv3.py", ["train", "valid"], ("train_vqa_inceptionv3_lstm.py", "train_vqa_inceptionv3_gru.py")),
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
for type1_file, args, type2_files in file_pairs:
    # Run the first type file with the first set of arguments
    print(f"Running {type1_file} with arguments {args}...")


    if run_script(type1_file, [args[0]]):
        print(f"Successfully ran {type1_file} with arguments {args[0]}.")

        print(f"Running {type1_file} with arguments {args[1]}...")
        if run_script(type1_file, [args[1]]):
            print(f"Successfully ran {type1_file} with arguments {args[1]}. Now running {type2_files}...")
            for type2_file in type2_files:
                if run_script(type2_file, []):
                    print(f"Successfully ran {type2_file}.")
                else:
                    print(f"Failed to run {type2_file}.")
                
        else:
            print(f"Failed to run {type1_file} with arguments {args[1]}.")
    else:
        print(f"Failed to run {type1_file} with arguments {args[0]}. Skipping subsequent runs.")