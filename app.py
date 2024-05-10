import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict car price
def predict_price(name, company, year, kms_driven, fuel_type):
    # Preprocess the input
    # You might need to preprocess the input data according to how your model was trained
    new_data = pd.DataFrame([[name, company,year , kms_driven,fuel_type ]], 
                        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    # Make prediction
    prediction = model.predict(new_data)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Car Price Prediction')

    # Form for user input
    name = st.text_input('Enter Car Name:')
    company = st.text_input('Enter Company Name:')
    year = st.number_input('Enter Year of Manufacture:', min_value=1900, max_value=2023, step=1)
    kms_driven = st.number_input('Enter Kilometers Driven:', min_value=0)
    fuel_type = st.selectbox('Select Fuel Type:', ['Petrol', 'Diesel', 'CNG'])

    if st.button('Predict'):
        if name and company and year and kms_driven and fuel_type:
            # Predict car price
            prediction = predict_price(name, company, year, kms_driven, fuel_type)
            st.success(f'Predicted Price: {prediction:.2f} INR')
        else:
            st.error('Please fill in all the input fields.')

if __name__ == '__main__':
    main()
