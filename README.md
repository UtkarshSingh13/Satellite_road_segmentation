# Satellite_road_segmentation
# Data Preprocessing
Due to the large size of the images (1024x1024 pixels), which resulted in each numpy array being around 9GB,making it infeasible to create or store the preprocessed data on Google Drive.  I converted the images to 256x256 numpy arrays to make them more manageable for training using a prepare_data.py script locally. Therefore, the images were downscaled to 256x256 pixels locally. The preprocessed train data files (train_xx.npy and train_yy.npy) were then uploaded for training. Patching was not used becuase the dataswet and .npy array would have been too large to manage(increase by 16 times)

# Model Architecture
The model used for this task is a U-Net with a ResNet-34 backbone. U-Net is a convolutional neural network architecture designed for fast and precise segmentation of images, which is ideal for our task of road extraction from satellite images.

# Training
The training was performed on Google Colab TPU due to the high RAM requirement (around 30GB). The training process was as follows:

Session 1: 12 epochs
Session 2: Additional 12 epochs using the previous checkpoint
Results
The accuracy achieved after the first 24 epochs was:

Training Accuracy: 98.53%
Validation Accuracy: 97.63%
Due to the TPU limit of 3 hours per day, I am submitting the 24-epoch result. The accuracy has not peaked and would likely increase with further training.
