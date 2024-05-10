import streamlit as st
import pickle
import numpy as np

# Load the ridge regressor and standard scaler pickle files
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function to preprocess input data
def preprocess_input(Temperature, RH, Ws, Rain, FFMC, DMC, ISI):
    new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI]])
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

    if st.button('Predict'):
        new_data_scaled = preprocess_input(Temperature, RH, Ws, Rain, FFMC, DMC, ISI)
        result = predict(new_data_scaled)
        st.write(f'FWI Score: {result}')
        if result<5:
            st.write('Type of Fire: Creeping surface fire')
            st.write("Potential Danger: Fire well be self extinguishing")
        elif result>=5 and result<10:
            st.write('Type of Fire: Low vigor surface fire')
            st.write("Potential Danger: Easily suppressed with hand tools")
        elif result>=10 and result<20:
            st.write('Type of Fire: Moderately vigorous surface fire')
            st.write("Potential Danger: Power pumps and hoses are required")
        elif result>=20 and result<30:
            st.write('Type of Fire: High intensity surface fire')
            st.write("Potential Danger: Difficult to control")
        else:
            st.write('Type of Fire: Very high intensity surface fire')
            st.write("Immediate and strong action is critical")

if __name__ == '__main__':
    main()
