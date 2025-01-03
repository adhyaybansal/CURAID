# CURAID
Curaid: Skin and Heart Disease Detection System
Curaid is a comprehensive system designed to predict skin and heart diseases using machine learning models. It also recommends pathology tests based on the predictions. The project integrates two key components:

# 1. Skin Disease Classification
Description: Predicts the type of skin disease from an uploaded image and recommends suitable pathology tests.
Features and Methods:
Dataset: DermNet dataset of skin disease images.
Model: Transfer learning using ResNet50 pre-trained on ImageNet.
Preprocessing:
Resizing images to 
224
×
224
224×224.
One-hot encoding for class labels.
Training:
Splitting data into 80% training and 20% validation sets.
Optimizer: Adam.
Loss function: Categorical crossentropy.
20 epochs with a batch size of 64.
Output: Saves the trained model as disease_classification_model.h5.
Additional Files: Exports label_encoder.pkl for decoding predictions.
