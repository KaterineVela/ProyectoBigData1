import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
modelo = joblib.load("modelo_entrenado.pkl")

# Título
st.title("Predicción de Enfermedad Cardíaca")
st.write("Introduce los datos del paciente para predecir el riesgo.")

# Crear entradas de usuario
gender = st.selectbox("Sexo", ["Hombre", "Mujer"])
smoker = st.selectbox("¿Fuma actualmente?", ["Sí", "No"])
bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, step=0.1)
physical_activity = st.selectbox("Actividad física frecuente", ["Sí", "No"])
age = st.number_input("Edad", min_value=18, max_value=100)
sleep_time = st.number_input("Horas de sueño promedio", min_value=0, max_value=24)

# Procesamiento de datos (ajusta según tu modelo)
input_data = pd.DataFrame({
    'BMI': [bmi],
    'Smoking': [1 if smoker == "Sí" else 0],
    'PhysicalActivity': [1 if physical_activity == "Sí" else 0],
    'Sex': [1 if gender == "Hombre" else 0],
    'AgeCategory': [age],
    'SleepTime': [sleep_time],
    # Agrega más columnas si tu modelo las necesita
})

# Botón de predicción
if st.button("Predecir riesgo"):
    prediccion = modelo.predict(input_data)[0]
    probabilidad = modelo.predict_proba(input_data)[0][1]

    if prediccion == 1:
        st.error(f"¡Riesgo alto de enfermedad cardíaca! (Probabilidad: {probabilidad:.2f})")
    else:
        st.success(f"Sin riesgo evidente de enfermedad cardíaca. (Probabilidad: {probabilidad:.2f})")
