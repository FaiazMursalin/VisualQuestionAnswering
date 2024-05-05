# Bridging the Gap: Uniting Information Retrieval and Visual Question


This project is a part of coursework for CSC790: Information Retrieval class offered in Spring 2024 at Missouri State University by Dr. Yassine Belkhouche.

## Inspiration
This class taught about information retreival systems, how to crawl data, create indexes, score and rank documents and perform topic modelling and text clustering. In addition to textual data, the real world also comprises of images. This project attempts to combine information retrieval and computer vision by exploring the research topic 'Visual Question Answering'.

## How to run

### Dataset
---
Dataset was obtained from: [Visual Question Answering](https://visualqa.org/download.html). Some minor directory modifications, such as renaming, moving json files out of a directory etc. were performed. The formatted dataset can be downloaded from here: [Formatted Dataset](https://livemissouristate-my.sharepoint.com/:u:/g/personal/dd932s_login_missouristate_edu/EZNm0xC4TUZEi_5UMxSCDHQBTtP2H0GSOIV2vNptyMjn0w?e=AO5eG1).

### Setting up the environment
---

The project directory consists of the following files:
- enviroment.yml
- requirement.txt

These 2 files can be used by ```conda``` or ```pip``` to setup the environment required to execute the codes in this project. The authors suggest the use of ```conda``` as the primary environment for this project. 

### Directory Setup
---

After collecting the dataset and extracting the dataset into a folder named `Data`, please run `1_directory_setup.py`. This script creates the neccessary subdirectories for saving the outputs.

### Image Feature Extraction
---

All python files matching `2_Image_feature_extraction*.py` are used for creating _"image vectors"_. Run them one by one after the directory setup is complete. Since both validation and training _"image vectors"_ are required, you will have to run each image extraction file twice with appropriate command line argument.

Example:

 ```python 2_Image_feature_extraction_autoencoder.py train```

  ```python 2_Image_feature_extraction_autoencoder.py valid```


### Text Vectorization and Training
---
All python files matching `3_train*.py` are used for extracting the text vectors and calculating correlation between text and _"image vectors"_. Run them one by one after the _"image vectors"_ for a particular architecture. 


## Not Important or maybe Most Important

Each individual file has comments describing what is happening in details. 
The `runall.py` script will run all the feature extraction and the training scripts one by one. But you will need to run `1_directory_setup.py` before you execute `runall.py`.

---
Disclaimer: Authors are aware that the code is not optimized and is full of redundancy. It will be fixed in future commits if either of them are suffering from over motivation and decides to question their entire life choice. 

