import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Page config
st.set_page_config(page_title="Depression Prediction", layout="centered")

st.title("Depression Prediction from Mental Health Survey")
st.write("Enter details and click Predict")

# ---------------- Model ----------------
# 1 = Depressed, 0 = Not Depressed

X_train = np.array([
    [1, 22, 6, 5, 7, 6, 5, 6, 1, 6],
    [0, 35, 2, 3, 8, 7, 7, 7, 0, 2],
    [1, 28, 7, 6, 6, 5, 4, 5, 1, 5],
    [0, 45, 1, 2, 7, 8, 8, 7, 0, 1],
])

y_train = np.array([1, 0, 1, 0])

model = Sequential([
    Dense(32, activation="relu", input_shape=(10,)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=30, verbose=0)

# ---------------- User Inputs (+ / - style) ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

age = st.number_input("Age", min_value=15, max_value=100, value=25, step=1)

academic_pressure = st.number_input("Academic Pressure (1–10)", 1, 10, 5)
work_pressure = st.number_input("Work Pressure (1–10)", 1, 10, 5)

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

study_satisfaction = st.number_input("Study Satisfaction (1–10)", 1, 10, 5)
job_satisfaction = st.number_input("Job Satisfaction (1–10)", 1, 10, 5)

sleep_duration = st.number_input("Sleep Duration (Hours)", 1, 12, 6)

suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts?", ["Yes", "No"]
)
suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0

financial_stress = st.number_input("Financial Stress (1–10)", 1, 10, 5)

# ---------------- Prediction ----------------
if st.button("Predict Depression"):
    user_input = np.array([[gender, age, academic_pressure, work_pressure,
                            cgpa, study_satisfaction, job_satisfaction,
                            sleep_duration, suicidal_thoughts, financial_stress]])

    prob = model.predict(user_input)[0][0]

    st.subheader("Prediction Result")

    if prob >= 0.5:
        st.error("Depression Detected")
    else:
        st.success("No Depression Detected")
