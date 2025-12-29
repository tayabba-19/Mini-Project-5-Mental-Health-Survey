import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Depression Prediction Fast", layout="centered")
st.title("🧠 Depression Prediction App (Fast)")

# Load CSV from GitHub
url = "https://raw.githubusercontent.com/tayabba-19/Mini-Project-5-Mental-Health-Survey/refs/heads/main/mental%20health%20survey.csv"
df = pd.read_csv(url)

# Clean column names
df.columns = [c.strip() for c in df.columns]

st.subheader("Dataset Preview")
st.dataframe(df.head())

target_column = "Depression"
# Ensure target is numeric 0/1
df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0).astype(int)

X = df.drop(columns=[target_column])
y = df[target_column]

# Fill missing & encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna("Unknown")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    else:
        X[col] = X[col].fillna(0)  # numeric missing → 0

# Prediction button
if st.button("Predict Depression"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    df["Predicted_Depression"] = predictions

    depressed_count = (predictions == 1).sum()
    not_depressed_count = (predictions == 0).sum()

    st.success("Prediction Completed")

    st.subheader("Prediction Summary")
    st.write(f"😟 Depressed (1): {depressed_count}")
    st.write(f"🙂 Not Depressed (0): {not_depressed_count}")

    # Pie chart
    st.subheader("Depression Distribution")
    fig, ax = plt.subplots()
    ax.pie([depressed_count, not_depressed_count],
           labels=["Depressed", "Not Depressed"],
           autopct='%1.1f%%', colors=['#FF4C4C','#4CAF50'])
    st.pyplot(fig)

    st.subheader("Sample Predictions")
    st.dataframe(df[[target_column, "Predicted_Depression"]].head(10))

    st.info("Label Meaning → 1 = Depressed | 0 = Not Depressed")
