import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from plotly.offline import iplot
import plotly as py
from sklearn.preprocessing import StandardScaler
import plotly.tools as tls
import cufflinks as cf
import plotly.figure_factory as ff
import plotly.express as px
import streamlit.components.v1 as components
import tensorflow as tf 


py.offline.init_notebook_mode(connected = True)
cf.go_offline()
cf.set_config_file(theme='white')

# Load SVM model from pickle file
with open("svm_model_new.pkl", "rb") as model_file:
    svc_model = pickle.load(model_file)

fires = pd.read_csv("forestfires.csv")

#changing days into numeric quantity because machine learning model deals with numbers
fires.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)

#changing month into numeric quantity
fires.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)


fires['area'].values[fires['area'].values > 0] = 1

#renaming the area attribute to output for clear understanding
fires = fires.rename(columns={'area': 'output'})

scaler = StandardScaler()
#fitting forest fire dataset to scaler by removing the attribute output
scaler.fit(fires.drop('output',axis=1))

scaled_features = scaler.transform(fires.drop('output',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=fires.columns[:-1])

X = df_feat.drop(columns=["X","Y","month","day"],axis=1)
y = fires['output']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35,random_state=200)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
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