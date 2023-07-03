# Visual Plant Disease Detection using Deep Learning, Machine learning and eXplainable AI Techniques
In this project, we investigate the potential of Artificial Intelligence to enhance identifications of plant diseases using Computer Vision and eXplainable AI (XAI). This project is the key component of the Multimodal Machine Learning course curriculum and is part of the Master's program at National Center for Scientific Research "Demokritos" and the University of Piraeus. Our approach involves comparing the results of Machine Learning and Deep Learning models. Additionally, we extract features from the plants using techniques such as Haralick, Gabor, shape and, histogram features which are used to train our Machine Learning models for classification. Moreover, by leveraging the power of XAI we aim to provide transparent and interpretable results to improve decision making in this task. Our results demonstrate the effectiveness of our approach in accurately detecting plant diseases and providing interpretable results.

### About Dataset
In this study, we used the New Plant Diseases Dataset for our computer vision task. The dataset consists of approximately 54.305 images of both healthy and inflected leaves of crop plants, divided into 38 distinct classes. Specifically, this dataset has been generated using offline augmentation techniques from the original PlantVillage Dataset. What's more, we created a new segmented dataset by applying a mask to existing images to remove the background and then we merged the two datasets together.


## Basic Code Information
The code has been developed in Kaggle & Google Colab. To run the code, the requirements.txt file must be used in order to load all the required modules/packages. 

## Repository Structure
The repository structure is simple and self-explanatory. It containts the following folders and files:

**Requirements Folder** - Contains the requirments file in .txt format.

**Presentation folder** - Contains the presentation in .pptx format.

**Report folder** - Contains the report as .pdf file.

**Data folder** - Contains a .txt file with the links to download the dataset, the extracted features and the saved models from Kaggle.

**Demo folder** - contains the following files
| Files/Folders    |  Description                         |              
|------------------|--------------------------------------|
| Demo-Menu.ipynb  | Main file for the demo code |
| EfficientNet_model.h5  | The saved EfficientNetB3 model |

**src folder** - contains the following files and folders
| Files/Folders         |  Description                         |              
|-----------------------|--------------------------------------|
| Visual_Plant_Disease_Detection_XAI.ipynb| Main file of our source code |