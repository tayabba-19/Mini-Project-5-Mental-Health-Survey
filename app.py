import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Depression Prediction", layout="centered")
st.title("🧠 Depression Prediction App")

# Load CSV from GitHub raw link
url = "https://raw.githubusercontent.com/tayabba-19/Mini-Project-5-Mental-Health-Survey/refs/heads/main/mental%20health%20survey.csv"
df = pd.read_csv(url)

st.subheader("Dataset Preview")
st.dataframe(df.head())

target_column = "Depression"  # 1 = Depressed, 0 = Not Depressed

# Clean column names (remove spaces)
df.columns = [c.strip() for c in df.columns]

X = df.drop(columns=[target_column])
y = df[target_column]

# Encode only categorical features (excluding target)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna("Unknown")  # fill missing
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train model
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
    st.write(f"😟 Depressed (1): {depressed_count} people")
    st.write(f"🙂 Not Depressed (0): {not_depressed_count} people")

    st.subheader("Sample Predictions")
    st.dataframe(df[[target_column, "Predicted_Depression"]].head(10))

    st.info("Label Meaning → 1 = Depressed | 0 = Not Depressed")
