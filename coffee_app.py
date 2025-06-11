
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load or initialize dataset
DATA_FILE = "user_brew_data.csv"
if os.path.exists(DATA_FILE):
    user_data = pd.read_csv(DATA_FILE)
else:
    user_data = pd.DataFrame(columns=['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s', 'TDS'])

# Define and train model from data
def train_model(data):
    X = data[['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s']]
    y = data['TDS']
    preprocessor = ColumnTransformer(
        transformers=[
            ('method', OneHotEncoder(), ['Method']),
            ('num', SimpleImputer(strategy='mean'), ['Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s'])
        ]
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

model = train_model(user_data)

st.title("☕ Coffee Brewing TDS Predictor")

# Mode selection
mode = st.radio("Select mode:", ["Predict TDS", "Suggest Parameters for Desired TDS", "Free Mode - Add New Data"])

if mode == "Predict TDS":
    method = st.selectbox("Brewing Method", ["V60", "Aeropress", "French Press", "Clever Dripper"])
    grind_size = st.slider("Grind Size (Microns)", 200, 1000, 400)
    temperature = st.slider("Water Temperature (°C)", 70, 100, 92)
    ratio = st.slider("Water-to-Coffee Ratio", 10.0, 22.0, 16.0, step=0.1)
    brew_time = st.number_input("Brew Time (seconds)", min_value=0, max_value=1000, value=90)

    if st.button("Predict TDS"):
        sample = pd.DataFrame({
            'Method': [method],
            'Grind_Size_Microns': [grind_size],
            'Temp_C': [temperature],
            'Ratio': [ratio],
            'Time_s': [brew_time]
        })
        prediction = model.predict(sample)[0]
        st.success(f"Predicted TDS: {prediction:.2f}%")

elif mode == "Suggest Parameters for Desired TDS":
    method = st.selectbox("Brewing Method", ["V60", "Aeropress", "French Press", "Clever Dripper"])
    ratio = st.slider("Water-to-Coffee Ratio", 10.0, 22.0, 16.0, step=0.1)
    desired_tds = st.slider("Desired TDS (%)", 0.8, 1.6, 1.3, step=0.01)
    max_suggestions = st.slider("Number of suggestions to show", 1, 5, 3)

    method_params = {
        "V60": (200, 600, 85, 96, 30, 300),
        "Aeropress": (250, 800, 85, 95, 30, 150),
        "French Press": (500, 1000, 85, 95, 240, 600),
        "Clever Dripper": (300, 700, 85, 95, 60, 300)
    }

    gmin, gmax, tmin, tmax, timemin, timemax = method_params[method]

    if st.button("Suggest Parameters"):
        suggestions = []
        for grind in range(gmin, gmax + 1, 50):
            for temp in range(tmin, tmax + 1):
                for time in range(timemin, timemax + 1, 30):
                    sample = pd.DataFrame({
                        'Method': [method],
                        'Grind_Size_Microns': [grind],
                        'Temp_C': [temp],
                        'Ratio': [ratio],
                        'Time_s': [time]
                    })
                    tds = model.predict(sample)[0]
                    diff = abs(tds - desired_tds)
                    suggestions.append((diff, grind, temp, time, tds))

        suggestions.sort(key=lambda x: x[0])
        for i, (diff, grind, temp, time, predicted_tds) in enumerate(suggestions[:max_suggestions], start=1):
            st.success(f"Suggestion #{i}:\nGrind Size: {grind}μm\nTemperature: {temp}°C\nTime: {time}s\nPredicted TDS: {predicted_tds:.2f}%")

elif mode == "Free Mode - Add New Data":
    st.subheader("Add New Brewing Data")
    method = st.selectbox("Brewing Method", ["V60", "Aeropress", "French Press", "Clever Dripper"])
    grind_size = st.number_input("Grind Size (Microns)", min_value=200, max_value=1000, value=400)
    temperature = st.number_input("Water Temperature (°C)", min_value=70, max_value=100, value=92)
    ratio = st.number_input("Water-to-Coffee Ratio", min_value=10.0, max_value=22.0, value=16.0, step=0.1)
    brew_time = st.number_input("Brew Time (seconds)", min_value=0, max_value=1000, value=90)
    tds = st.number_input("Measured TDS (%)", min_value=0.5, max_value=2.0, value=1.3, step=0.01)

    if st.button("Submit New Data"):
        new_entry = pd.DataFrame({
            'Method': [method],
            'Grind_Size_Microns': [grind_size],
            'Temp_C': [temperature],
            'Ratio': [ratio],
            'Time_s': [brew_time],
            'TDS': [tds]
        })
        user_data = pd.concat([user_data, new_entry], ignore_index=True)
        user_data.to_csv(DATA_FILE, index=False)
        st.success("New data saved and included in future model predictions.")
