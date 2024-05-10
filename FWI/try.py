import tensorflow as tf 
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#loading the algerian_forest dataset into data frame using pandas
df = pd.read_csv("C:/Users/Sarthak Aggarwal/Downloads/Flare-Scan-main (1)/Flare-Scan-main/FWI_Predictor_ML/dataset/Algerian_forest_fires_cleaned_dataset.csv")

# Drop "day", "month", "year", and "Region" columns
df.drop(["day", "month", "year", "Region"], inplace=True, axis=1)

# Strip column names of any leading/trailing whitespace
df.columns = df.columns.str.strip()

# Separate features and target
X = df.drop(["FWI", "Classes"], axis=1)
y = df["FWI"]
z = df["Classes"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Drop "BUI" and "DC" columns after splitting
X_train.drop(["BUI", "DC"], axis=1, inplace=True)
X_test.drop(["BUI", "DC"], axis=1, inplace=True)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

ridgecv = RidgeCV(cv=5)
ridgecv.fit(X_train_scaled, y_train)
y_pred = ridgecv.predict(X_test_scaled)
print(X_train_scaled.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model (replace X_train, y_train with your actual data)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained TensorFlow model
model.save('tf_model')

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
