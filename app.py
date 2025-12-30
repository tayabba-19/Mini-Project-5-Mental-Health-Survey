import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- Page Config ----------------
st.set_page_config(page_title="Depression Prediction App", layout="centered")

st.title("Depression Prediction from Mental Health Survey")
st.write("Enter details and click Predict to check depression probability")

# ---------------- Deep Learning Model ----------------
# Binary Classification
# 1 = Depressed, 0 = Not Depressed

# Dummy training data (for deployment & evaluation)
X_train = np.array([
    [1, 22, 6, 5, 7, 6, 5, 6, 1, 6],
    [0, 35, 2, 3, 8, 7, 7, 7, 0, 2],
    [1, 28, 7, 6, 6, 5, 4, 5, 1, 5],
    [0, 45, 1, 2, 7, 8, 8, 7, 0, 1],
    [1, 30, 6, 5, 6, 6, 5, 6, 1, 5],
    [0, 50, 1, 2, 8, 7, 7, 8, 0, 2]
])

y_train = np.array([1, 0, 1, 0, 1, 0])

model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(10,)))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=40, verbose=0)

# ---------------- User Inputs (CSV-based) ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

age = st.number_input("Age", min_value=15, max_value=100, value=25)

academic_pressure = st.slider("Academic Pressure (1–10)", 1, 10, 5)
work_pressure = st.slider("Work Pressure (1–10)", 1, 10, 5)

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)

study_satisfaction = st.slider("Study Satisfaction (1–10)", 1, 10, 5)
job_satisfaction = st.slider("Job Satisfaction (1–10)", 1, 10, 5)

sleep_duration = st.slider("Sleep Duration (Hours)", 1, 12, 6)

suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts?", ["Yes", "No"]
)
suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0

financial_stress = st.slider("Financial Stress (1–10)", 1, 10, 5)

# ---------------- Prediction ----------------
if st.button("Predict Depression"):
    user_input = np.array([[gender, age, academic_pressure, work_pressure,
                            cgpa, study_satisfaction, job_satisfaction,
                            sleep_duration, suicidal_thoughts, financial_stress]])

    prediction = model.predict(user_input)

    depressed = prediction[0][0] * 100
    not_depressed = 100 - depressed

    st.subheader("Prediction Result")
    st.success(f"Depressed Probability: {depressed:.2f}%")
    st.info(f"Not Depressed Probability: {not_depressed:.2f}%")
