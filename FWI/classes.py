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

from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()

# Apply label encoding to the 'Sex' column
df['Classes_encoded'] =1- le.fit_transform(df['Classes'])


df.drop(["Classes"],inplace=True,axis=1)

df.rename(columns={'Classes_encoded': 'Classes'}, inplace=True)

X = df.drop(["FWI", "Classes"], axis=1)
y = df["FWI"]
z = df["Classes"]


X_train,X_test,z_train,z_test = train_test_split(X,z,test_size=0.25,random_state = 42)
X_train.drop(["BUI","DC"],axis = 1,inplace = True)
X_test.drop(["BUI","DC"],axis = 1,inplace = True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
logistic_model = LogisticRegression()
logistic_model.fit(X_train,z_train)

predictions = logistic_model.predict(X_test)


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
model.fit(X_train, z_train, epochs=10, batch_size=32)

# Save the trained TensorFlow model
model.save('tf_model_cls')

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('tf_model_cls')
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('model_cls.tflite', 'wb') as f:
    f.write(tflite_model)