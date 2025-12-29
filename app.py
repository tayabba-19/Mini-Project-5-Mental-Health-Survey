import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Depression Prediction", layout="centered")

st.title("🧠 Depression Prediction App")
st.write("Mental Health Survey Dataset (Auto Loaded)")

# 🔹 Auto load CSV from GitHub repo
df = pd.read_csv("mental_health_survey.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Target column
target_column = "Depression"   # 1 = Depressed, 0 = Not Depressed

# Features & target
X = df.drop(columns=[target_column])
y = df[target_column]

# Encoding categorical columns
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 🔘 Prediction Button
if st.button("Predict Depression"):
    predictions = model.predict(X)
    df["Predicted_Depression"] = predictions

    # Count results
    depressed_count = (predictions == 1).sum()
    not_depressed_count = (predictions == 0).sum()

    st.success("Prediction Completed")

    st.subheader("Prediction Summary")
    st.write(f"😟 **Depressed (1): {depressed_count} people**")
    st.write(f"🙂 **Not Depressed (0): {not_depressed_count} people**")

    st.subheader("Prediction Output (Sample)")
    st.dataframe(df[[target_column, "Predicted_Depression"]].head(10))

    st.info("Label Meaning → 1 = Depressed | 0 = Not Depressed")
