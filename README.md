# CURAID: Skin and Heart Disease Detection System

Curaid is a comprehensive machine learning-based system designed to predict skin and heart diseases and recommend relevant pathology tests. The project integrates two key components: Skin Disease Classification and Heart Disease Prediction.

## 1. Skin Disease Classification

### Description:
This component predicts the type of skin disease from an uploaded image and recommends suitable pathology tests.

### Features and Methods:
- **Dataset**: DermNet dataset of skin disease images.
- **Model**: Transfer learning using ResNet50, pre-trained on ImageNet.
- **Preprocessing**:
  - Resizing images to 224 × 224 pixels.
  - One-hot encoding for class labels.
- **Training**:
  - Data split into 80% training and 20% validation sets.
  - Optimizer: Adam.
  - Loss function: Categorical Crossentropy.
  - 20 epochs with a batch size of 64.
- **Output**: Saves the trained model as `disease_classification_model.h5`.
- **Additional Files**: Exports `label_encoder.pkl` for decoding predictions.

## 2. Heart Disease Prediction

### Description:
This component predicts the likelihood of heart disease based on user inputs such as age, blood pressure, cholesterol, smoking habits, etc., and suggests relevant pathology tests.

### Features and Methods:
- **Dataset**: Cardiovascular disease dataset.
- **Preprocessing**:
  - Standardizing features like age, weight, and height.
  - Encoding categorical variables (cholesterol, glucose).
  - Removing outliers and unnecessary columns.
- **Model**: Random Forest Classifier with 100 estimators.
- **Training**:
  - Data split into 80% training and 20% testing sets.
  - Cross-validation with 5 folds.
  - Metrics: Accuracy, classification report, and confusion matrix.
- **Output**: Saves the trained model as `random_forest_model.joblib`.
- **Additional Files**: Exports `cardio_label_encoder.pkl` for encoding categorical features.

## Deployment
The project includes a dummy menu section simulating a web interface. The system is further deployed on a website by a web developer.

## Technologies Used

- **Libraries**:
  - Python: NumPy, Pandas, TensorFlow, Keras, Scikit-learn, OpenCV, Joblib.
  - Data visualization: Matplotlib.
- **Models**:
  - ResNet50 for skin disease classification.
  - Random Forest for heart disease prediction.
- **Preprocessing Tools**: Label encoding, one-hot encoding, standardization.


![image](https://github.com/user-attachments/assets/1bd85235-c5c5-41d5-9a84-8c304f34e875)
![image](https://github.com/user-attachments/assets/d5a6b4b7-7561-4e9d-a6ba-fe66f305e0b6)
![image](https://github.com/user-attachments/assets/e866cff3-ae4e-4a02-a419-607064e51337)
![image](https://github.com/user-attachments/assets/bfc58f3a-ea72-4c07-bf53-d912c0d51ffd)




