# CURAID
Curaid: Skin and Heart Disease Detection System
Curaid is a comprehensive system designed to predict skin and heart diseases using machine learning models. It also recommends pathology tests based on the predictions. The project integrates two key components:

1. Skin Disease Classification
Description: Predicts the type of skin disease from an uploaded image and recommends suitable pathology tests.
Features and Methods:
  Dataset: DermNet dataset of skin disease images.
  Model: Transfer learning using ResNet50 pre-trained on ImageNet.
  Preprocessing:
    Resizing images to 224 × 224
    One-hot encoding for class labels.
Training:
Splitting data into 80% training and 20% validation sets.
Optimizer: Adam.
Loss function: Categorical crossentropy.
20 epochs with a batch size of 64.
Output: Saves the trained model as disease_classification_model.h5.
Additional Files: Exports label_encoder.pkl for decoding predictions.
3. Heart Disease Prediction
Description: Predicts the likelihood of heart disease based on user inputs like age, blood pressure, cholesterol, smoking habits, etc., and suggests relevant tests.
Features and Methods:
Dataset: Cardiovascular disease dataset.
Preprocessing:
Standardizing features such as age, weight, and height.
Encoding categorical variables (cholesterol, glucose).
Removing outliers and unnecessary columns.
Model: Random Forest Classifier with 100 estimators.
Training:
Splitting data into training (80%) and testing (20%) sets.
Cross-validation with 5 folds.
Metrics: Accuracy, classification report, and confusion matrix.
Output: Saves the trained model as random_forest_model.joblib.
Additional Files: Exports cardio_label_encoder.pkl for encoding categorical features.
Deployment
The project includes a dummy menu section within the code, simulating a web interface. It was further deployed on a website by a web developer.

Technologies Used
Libraries:
Python: NumPy, Pandas, TensorFlow, Keras, Scikit-learn, OpenCV, Joblib.
Data visualization: Matplotlib.
Models:
ResNet50 for skin disease classification.
Random Forest for heart disease prediction.
Preprocessing Tools: Label encoding, one-hot encoding, standardization.
