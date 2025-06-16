
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Coffee TDS App", layout="wide")

# Load image logo
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.svg/2048px-A_small_cup_of_coffee.svg.png", width=80)

# Language selector
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"])
texts = {
    "title": {"English": "☕ Coffee Brewing TDS Calculator", "Español": "☕ Calculadora de TDS para Café"},
    "select_mode": {"English": "Select mode:", "Español": "Selecciona el modo:"},
    "predict": {"English": "Predict TDS", "Español": "Predecir TDS"},
    "suggest": {"English": "Suggest Parameters", "Español": "Sugerir Parámetros"},
    "free": {"English": "Free Mode - Add Data", "Español": "Modo Libre - Añadir Datos"},
    "dashboard": {"English": "Saved Brews Dashboard", "Español": "Panel de Brews Guardados"}
}

DATA_FILE = "user_brew_data.csv"
if os.path.exists(DATA_FILE):
    user_data = pd.read_csv(DATA_FILE)
else:
    user_data = pd.DataFrame(columns=['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s', 'TDS', 'Tags'])

def train_model(data):
    if data.empty:
        return None
    X = data[['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s']]
    y = data['TDS']
    preprocessor = ColumnTransformer([
        ('method', OneHotEncoder(), ['Method']),
        ('num', SimpleImputer(), ['Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s'])
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

model = train_model(user_data)

st.title(texts["title"][lang])
mode = st.sidebar.radio(texts["select_mode"][lang], [
    texts["predict"][lang], texts["suggest"][lang],
    texts["free"][lang], texts["dashboard"][lang]
])

if model is None and mode != texts["free"][lang]:
    st.warning("Please add some data first using Free Mode.")

if mode == texts["predict"][lang] and model:
    method = st.selectbox("Method", ["V60", "Aeropress", "French Press", "Clever Dripper"])
    grind = st.slider("Grind Size (μm)", 200, 1000, 500)
    temp = st.slider("Temperature (°C)", 70, 100, 93)
    ratio = st.slider("Ratio", 10.0, 22.0, 16.0, step=0.1)
    time = st.slider("Brew Time (s)", 30, 600, 150)
    if st.button("Predict"):
        input_df = pd.DataFrame([[method, grind, temp, ratio, time]], columns=['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s'])
        pred = model.predict(input_df)[0]
        st.success(f"Predicted TDS: {pred:.2f}%")

if mode == texts["suggest"][lang] and model:
    method = st.selectbox("Method", ["V60", "Aeropress", "French Press", "Clever Dripper"])
    ratio = st.slider("Ratio", 10.0, 22.0, 16.0, step=0.1)
    desired_tds = st.slider("Desired TDS", 0.8, 1.6, 1.3)
    suggestions = []
    for g in range(300, 800, 100):
        for t in range(85, 96, 2):
            for b in range(60, 300, 60):
                sample = pd.DataFrame([[method, g, t, ratio, b]], columns=['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s'])
                predicted = model.predict(sample)[0]
                suggestions.append((abs(predicted - desired_tds), g, t, b, predicted))
    suggestions.sort()
    for i, (diff, g, t, b, pred) in enumerate(suggestions[:3]):
        st.info(f"Grind: {g}μm | Temp: {t}°C | Time: {b}s → TDS: {pred:.2f}%")

if mode == texts["free"][lang]:
    st.subheader("New Brew Entry")
    method = st.selectbox("Method", ["V60", "Aeropress", "French Press", "Clever Dripper"])
    grind = st.number_input("Grind Size (μm)", 200, 1000, 500)
    temp = st.number_input("Temperature (°C)", 70, 100, 93)
    ratio = st.number_input("Ratio", 10.0, 22.0, 16.0)
    time = st.number_input("Brew Time (s)", 30, 600, 150)
    tds = st.number_input("Measured TDS", 0.8, 1.6, 1.35)
    tags = st.text_input("Tags (comma separated)", "lightroast,ethiopia")
    if st.button("Submit"):
        new_row = pd.DataFrame([[method, grind, temp, ratio, time, tds, tags]],
                               columns=['Method', 'Grind_Size_Microns', 'Temp_C', 'Ratio', 'Time_s', 'TDS', 'Tags'])
        user_data = pd.concat([user_data, new_row], ignore_index=True)
        user_data.to_csv(DATA_FILE, index=False)
        st.success("Saved successfully!")

if mode == texts["dashboard"][lang]:
    st.subheader("📋 Brew Dashboard")
    search = st.text_input("Search by tag or method")
    filtered = user_data.copy()
    if search:
        filtered = filtered[filtered.apply(lambda row: search.lower() in row.astype(str).str.lower().str.cat(sep=' '), axis=1)]
    st.dataframe(filtered)
    if st.checkbox("Show TDS Chart"):
        st.line_chart(filtered[['TDS']])
