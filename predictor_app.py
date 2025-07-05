
import streamlit as st
import pandas as pd
import joblib

# Load your trained model
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("E0.csv")

# Prepare data
features = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
X = df[features]
y_result = df["FTR"]
y_over = df["FTHG"] + df["FTAG"]
y_over = y_over.apply(lambda x: "Over" if x > 2.5 else "Under")

# Train models
model_result = RandomForestClassifier()
model_result.fit(X, y_result)

model_over = RandomForestClassifier()
model_over.fit(X, y_over)

# UI
st.title("⚽ Football Match Predictor")
st.write("Enter the match details below to predict the result and over/under 2.5 goals.")

# Inputs
fthg = st.number_input("Enter predicted Home Goals", min_value=0, step=1)
ftag = st.number_input("Enter predicted Away Goals", min_value=0, step=1)
hs = st.number_input("Enter Home Shots", min_value=0, step=1)
as_ = st.number_input("Enter Away Shots", min_value=0, step=1)
hst = st.number_input("Enter Home Shots on Target", min_value=0, step=1)
ast = st.number_input("Enter Away Shots on Target", min_value=0, step=1)
hf = st.number_input("Enter Home Fouls", min_value=0, step=1)
af = st.number_input("Enter Away Fouls", min_value=0, step=1)
b365h = st.number_input("Enter B365 Home Win Odds", min_value=0.0, step=0.1)
b365d = st.number_input("Enter B365 Draw Odds", min_value=0.0, step=0.1)
b365a = st.number_input("Enter B365 Away Win Odds", min_value=0.0, step=0.1)
b365o = st.number_input("Enter B365 Over 2.5 Odds", min_value=0.0, step=0.1)
b365u = st.number_input("Enter B365 Under 2.5 Odds", min_value=0.0, step=0.1)

if st.button("🔮 Predict"):
    input_data = [[fthg, ftag, hs, as_, hst, ast, hf, af, b365h, b365d, b365a, b365o, b365u]]
    result = model_result.predict(input_data)[0]
    over = model_over.predict(input_data)[0]

    st.subheader("📈 Prediction Results")
    st.success(f"🏆 Match Result: {result}")
    st.info(f"📊 Over/Under 2.5: {over}")
