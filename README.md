# Histopathologic Cancer Detection using Deep Learning Methods

This project focuses on developing deep learning models to accurately classify histopathologic cancerous images as cancer (0) or benign (1). The goal is to provide a faster and more accurate method for earlier diagnoses and better treatment outcomes for patients.

## Dataset

The dataset used for this project is the Pcam dataset, which consists of 220,000 training images and 60,000 test images. Each image has a shape of (96 x 96 x 3), representing 96 pixels in the x-axis, 96 pixels in the y-axis, and 3 RGB color values. The training set has a ratio of 60% benign (1) and 40% cancer (0), eliminating the need to balance the dataset. Most of the images are colored, with only a few being black and white. These noise data have minimal contribution to the loss function and can be ignored.

## Data Preparation

The following steps were performed for data preprocessing:

- Normalization of pixel values: The pixel values of the images were normalized to range between 0 and 1.
- Data Augmentation: As the cancer cells are detected at the center of the image, experiments were conducted with and without zooming by reducing the image size from 96 x 96 (px) to 48 x 48 (px).
- Batches and Generators: Batches of data were created with a batch size of 256 images per batch. Generators were used to feed the training data to the model.
- Training and Validation Split: The dataset was split into a training set (75%) and a validation set (25%).

## Baseline Model Architecture

The baseline model used for training is a deep learning CNN model with 8 hidden layers and 1 output layer. The architecture of the baseline model is as follows:

- 4 Convolutional layers with a kernel size of (3 x 3), a stride of 1, and the activation function 'relu'.
- 2 Max pooling layers following the convolutional layers, with a pooling shape of (2 x 2).
- 1 Flatten layer to convert the images to 1-D vectors.
- 1 Dense layer with 256 units and the activation function 'relu'.
- 1 Output Dense layer with 1 unit and the activation function 'sigmoid'.

The baseline model has a total of 680,721 trainable parameters.

## Baseline Model Performance

The baseline model was trained with the following configuration:

- Loss Function: Binary Cross Entropy
- Optimizer: Adam with a learning rate of 0.001
- Training on 10 epochs

## Improved Model Architecture

An improved model was developed using transfer learning with the VGG19 model and trainable base model parameters. The architecture of the improved model is as follows:

- VGG19 model as the base model with pre-trained weights.
- 16 Convolutional layers with a kernel size of (3 x 3), a stride of 1, and the activation function 'relu'.
- 5 Max pooling layers with a pooling shape of (2 x 2).
- 1 Flatten layer to convert the input to a 1-D vector.
- 2 fully connected Dense layers.
- 2 BatchNormalization layers and 2 dropout layers with a dropout rate of 0.3.
- 1 Output Dense layer with 1 unit and the activation function 'sigmoid'.

The improved model has a total of 20 million trainable parameters.

## Improved Model Performance

The improved model was trained with the following configuration:

- Loss Function: Binary Cross Entropy
- Optimizer: Adam with a learning rate of 0.001
- Training on 20 epochs

In addition to training and validation accuracy and loss

, the Area Under the Receiver Operating Characteristic (AUC-ROC) curve was also evaluated for the improved model.

## Model Comparison

The baseline CNN model with zooming had average accuracies. However, the same CNN model without zooming achieved higher accuracy. Transfer learning with the VGG16 model without trainable base parameters showed similar accuracy, while transfer learning with the VGG19 model with trainable base parameters achieved the highest accuracy.

## Conclusion

In this project, a baseline CNN model was used, and transfer learning with a trainable base model was implemented to improve the accuracy and training speed. The improved model showed improved training/validation accuracy, faster training time, and no overfitting. Minor misclassifications were observed with the improved model. The results demonstrate the potential of deep learning methods in histopathologic cancer detection, providing a valuable tool for early diagnoses and improved treatment outcomes.

Kaggle competition link :
https://www.kaggle.com/c/histopathologic-cancer-detection
