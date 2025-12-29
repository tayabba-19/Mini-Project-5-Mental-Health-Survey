import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Depression Prediction Fast", layout="centered")
st.title("🧠 Depression Prediction App (Fast)")

# Load CSV automatically from GitHub
url = "https://raw.githubusercontent.com/tayabba-19/Mini-Project-5-Mental-Health-Survey/refs/heads/main/mental%20health%20survey.csv"
df = pd.read_csv(url)
df.columns = [c.strip() for c in df.columns]

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features & Target
target_column = "Depression"  # 1=Depressed, 0=Not Depressed
X = df.drop(columns=[target_column])
y = df[target_column].astype(int)

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna("Unknown")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 🔘 Prediction Button
if st.button("Predict Depression"):
    # Train & predict instantly
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    df["Predicted_Depression"] = predictions

    # Count
    depressed_count = (predictions == 1).sum()
    not_depressed_count = (predictions == 0).sum()

    st.success("Prediction Completed")

    # Show counts
    st.subheader("Prediction Summary")
    st.write(f"😟 Depressed (1): {depressed_count} people")
    st.write(f"🙂 Not Depressed (0): {not_depressed_count} people")

    # Pie chart
    st.subheader("Depression Distribution")
    fig, ax = plt.subplots()
    ax.pie([depressed_count, not_depressed_count], labels=["Depressed", "Not Depressed"],
           autopct='%1.1f%%', colors=['#FF4C4C','#4CAF50'])
    st.pyplot(fig)

    # Sample table
    st.subheader("Sample Predictions")
    st.dataframe(df[[target_column, "Predicted_Depression"]].head(10))

    st.info("Label Meaning → 1 = Depressed | 0 = Not Depressed")
