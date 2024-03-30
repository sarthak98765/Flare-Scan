import streamlit as st
import pickle
import numpy as np

# Load the ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input(Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region):
    new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    return new_data_scaled

# Function to make predictions
def predict(new_data_scaled):
    result = ridge_model.predict(new_data_scaled)
    return result[0]

# Streamlit app
def main():
    st.title('Forest Fire Prediction')

    Temperature = st.number_input('Temperature')
    RH = st.number_input('Relative Humidity (RH)')
    Ws = st.number_input('Wind Speed (Ws)')
    Rain = st.number_input('Rain')
    FFMC = st.number_input('Fine Fuel Moisture Code (FFMC)')
    DMC = st.number_input('Duff Moisture Code (DMC)')
    ISI = st.number_input('Initial Spread Index (ISI)')
    Classes = st.number_input('Classes')
    Region = st.number_input('Region')

    if st.button('Predict'):
        new_data_scaled = preprocess_input(Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region)
        result = predict(new_data_scaled)
        st.write(f'Predicted Result: {result}')

if __name__ == '__main__':
    main()
