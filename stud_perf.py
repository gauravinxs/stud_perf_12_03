import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model only once (faster)
# -----------------------------
@st.cache_resource
def load_model():
    with open("student_lr_model.pkl", "rb") as f:
        model, scaler, le = pickle.load(f)
    return model, scaler, le


# -----------------------------
# Preprocess input
# -----------------------------
def preprocess_input(data, scaler, le):

    # Convert dictionary to dataframe
    df = pd.DataFrame([data])

    # Encode categorical column
    df["Extracurricular Activities"] = le.transform(
        df["Extracurricular Activities"]
    )

    # Scale numerical values
    df_scaled = scaler.transform(df)

    return df_scaled


# -----------------------------
# Prediction function
# -----------------------------
def predict(data):

    model, scaler, le = load_model()

    processed_data = preprocess_input(data, scaler, le)

    prediction = model.predict(processed_data)

    return prediction[0]


# -----------------------------
# Streamlit UI
# -----------------------------
def main():

    st.title("📊 Student Performance Predictor")

    st.write("Enter student details to predict exam score")

    # User inputs
    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=12,
        value=5
    )

    previous_scores = st.number_input(
        "Previous Score",
        min_value=0,
        max_value=100,
        value=70
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0,
        max_value=12,
        value=6
    )

    sample_papers = st.number_input(
        "Sample Papers Practiced",
        min_value=0,
        max_value=20,
        value=5
    )

    # Button
    if st.button("Predict Score"):

        user_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": sample_papers
        }

        result = predict(user_data)

        st.success(f"Predicted Student Score: {result:.2f}")


# Run app
if __name__ == "__main__":
    main()