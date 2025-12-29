import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Depression Prediction Fast", layout="centered")
st.title("🧠 Depression Prediction App (Exact % Match)")

# Load CSV from GitHub
url = "https://raw.githubusercontent.com/tayabba-19/Mini-Project-5-Mental-Health-Survey/refs/heads/main/mental%20health%20survey.csv"
df = pd.read_csv(url)

# Clean column names
df.columns = [c.strip() for c in df.columns]

st.subheader("Dataset Preview")
st.dataframe(df.head())

target_column = "Depression"

# 🔹 Drop rows where target is NaN to match Colab
df = df.dropna(subset=[target_column])
df[target_column] = df[target_column].astype(int)  # ensure 0/1 exact

# Features & Target
X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna("Unknown")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train model once
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediction button
if st.button("Predict Depression"):
    predictions = model.predict(X)
    df["Predicted_Depression"] = predictions

    depressed_count = (predictions == 1).sum()
    not_depressed_count = (predictions == 0).sum()

    st.success("Prediction Completed")

    st.subheader("Prediction Summary")
    st.write(f"😟 Depressed (1): {depressed_count}")
    st.write(f"🙂 Not Depressed (0): {not_depressed_count}")

    # 🔹 Pie chart with exact percentage
    st.subheader("Depression Distribution")
    fig, ax = plt.subplots()
    total = depressed_count + not_depressed_count
    ax.pie([depressed_count, not_depressed_count],
           labels=["Depressed", "Not Depressed"],
           autopct=lambda p: f'{p*total/100:.1f}%',  # exact fraction based on total rows
           colors=['#FF4C4C','#4CAF50'])
    st.pyplot(fig)

    st.subheader("Sample Predictions")
    st.dataframe(df[[target_column, "Predicted_Depression"]].head(10))

    st.info("Label Meaning → 1 = Depressed | 0 = Not Depressed")
