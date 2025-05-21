import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
heart_data = pd.read_csv('heart.csv')

# Preparing data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the following health parameters to predict:")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200)
chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=100, max_value=600)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting ECG Results (restecg)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST depression (oldpeak)', min_value=0.0, max_value=10.0, format="%.1f")
slope = st.selectbox('Slope of peak exercise ST segment (slope)', [0, 1, 2])
ca = st.selectbox('Number of major vessels (ca)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2])

# Convert sex to numeric
sex = 1 if sex == 'Male' else 0

# Prediction
if st.button('Predict'):
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                           oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ The person has heart disease.")
    else:
        st.success("✅ The person does NOT have heart disease.")

# Show model accuracy
st.markdown(f"**Training Accuracy:** {accuracy_score(model.predict(X_train), Y_train) * 100:.2f}%")
st.markdown(f"**Test Accuracy:** {accuracy_score(model.predict(X_test), Y_test) * 100:.2f}%")

# Footer
st.markdown(
    """
    <hr style="border:1px solid #eee;margin-top:2em;margin-bottom:1em"/>
    <div style='text-align: center;'>
        <span style='font-size: 18px;'>❤️ Made with Love by 
        <a href="https://github.com/kaabilcoder" target="_blank" style='text-decoration: none; color: #ff4b4b;'>kaabilcoder</a>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

