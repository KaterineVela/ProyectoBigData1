import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("modelo_ligero.pkl")

# Título de la app
st.title("Predicción de Enfermedad Cardíaca")

# Formulario de entrada de datos
st.subheader("Ingrese los datos del paciente")

# Crear inputs para cada característica (ejemplo simple)
gender = st.selectbox("¿Sexo masculino?", ["No", "Sí"])  # 1 o 0
age = st.slider("Edad", 18, 100, 50)
bmi = st.number_input("IMC (BMI)", 10.0, 60.0, 25.0)
smoking = st.selectbox("¿Fuma actualmente?", ["No", "Sí"])

# Convertir entradas a formato del modelo
input_dict = {
    "HighBP": 0,
    "HighChol": 0,
    "CholCheck": 1,
    "BMI": bmi,
    "Smoker": 1 if smoking == "Sí" else 0,
    "Stroke": 0,
    "Diabetes": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 1 if gender == "Sí" else 0,
    "Age": age,
    "Education": 4,
    "Income": 5,
}

input_df = pd.DataFrame([input_dict])

# Botón para predecir
if st.button("Predecir"):
    pred = modelo.predict(input_df)[0]
    st.success("✅ Riesgo de enfermedad cardíaca" if pred == 1 else "💚 Sin riesgo identificado")
