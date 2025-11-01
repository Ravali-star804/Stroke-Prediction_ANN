🧠 Stroke Prediction using Artificial Neural Network (ANN)
📘 Overview

This project focuses on predicting the likelihood of a stroke in patients based on various health parameters using an Artificial Neural Network (ANN).
The aim is to assist healthcare professionals in early diagnosis and prevention by analyzing patient data effectively.

🎯 Objective

To build and train a neural network model that can predict whether a person is likely to have a stroke based on clinical and demographic data.

📊 Dataset

Dataset Source: Stroke Prediction Dataset - Kaggle

Attributes Used:

gender

age

hypertension

heart_disease

ever_married

work_type

Residence_type

avg_glucose_level

bmi

smoking_status

stroke (Target Variable)

⚙️ Project Workflow
1. Data Preprocessing

Handled missing values in BMI.

Performed label encoding and one-hot encoding for categorical variables.

Normalized numerical features for better ANN convergence.

2. Model Building

Used Keras (TensorFlow backend) to build a Sequential ANN model.

Layers:

Input Layer (based on number of features)

Hidden Layers with ReLU activation

Output Layer with Sigmoid activation (binary classification)

3. Model Compilation

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

4. Model Training

Split the dataset into training and testing sets.

Trained the ANN for several epochs, observing loss and accuracy trends.

5. Evaluation

Evaluated model using:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Visualized training and validation accuracy/loss using Matplotlib.

🧮 Libraries Used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report

📈 Results

The model achieved good accuracy on the test data.

The ANN was able to identify key contributing factors to stroke risk.

Accuracy and performance depend on preprocessing and parameter tuning.

💡 Future Enhancements

Apply SMOTE or other resampling techniques to handle class imbalance.

Author

Ravali – Data Science Intern @ Rubixe
[LinkedIn](www.linkedin.com/in/ravali-ambadi-872772187)

Experiment with Deep Neural Networks or ensemble models for higher accuracy.

Deploy the model using Streamlit or Flask for real-time predictions.
