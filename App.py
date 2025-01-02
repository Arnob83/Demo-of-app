import sqlite3
import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import requests

# URL to the raw xgb_model_new.pkl file in your GitHub repository
url = "https://raw.githubusercontent.com/Arnob83/Demo-of-app/main/xgb_model_new.pkl"

# Download the xgb_model_new.pkl file and save it locally
response = requests.get(url)
with open("xgb_model_new.pkl", "wb") as file:
    file.write(response.content)

# Load the trained model
with open("xgb_model_new.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

# Database setup
def create_database():
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    # Create table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Credit_History TEXT,
            Education_1 TEXT,
            ApplicantIncome REAL,
            CoapplicantIncome REAL,
            Loan_Amount_Term REAL,
            Prediction TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_into_database(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Prediction):
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_inputs (
            Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Prediction
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Prediction))
    conn.commit()
    conn.close()

@st.cache_data
def prediction(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Convert user input
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data (column order doesn't matter here)
    input_data = pd.DataFrame(
        [[Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Ensure column order matches the classifier’s expectations
    expected_order = classifier.feature_names_in_
    input_data = input_data[expected_order]  # Reorder columns programmatically

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = classifier.predict(input_data)
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data

def explain_prediction(input_data, final_result):
    """
    Analyze features and provide a detailed explanation of the prediction,
    along with a bar chart for SHAP values.
    """
    # Initialize SHAP explainer specifically for XGBoost
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_data)

    # Extract SHAP values for the input data
    shap_values_for_input = shap_values[0]  # SHAP values for the first row of input_data

    # Prepare feature importance data
    feature_names = input_data.columns

    explanation_text = f"**Why your loan is {final_result}:**\n\n"
    for feature, shap_value in zip(feature_names, shap_values_for_input):
        explanation_text += (
            f"- **{feature}**: {'Positive' if shap_value > 0 else 'Negative'} contribution with a SHAP value of {shap_value:.2f}\n"
        )

    # Identify the main factors contributing to the decision
    if final_result == 'Rejected':
        explanation_text += "\nThe loan was rejected because the negative contributions outweighed the positive ones."
    else:
        explanation_text += "\nThe loan was approved because the positive contributions outweighed the negative ones."

    # Create bar chart for SHAP values
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_values_for_input, color=["green" if val > 0 else "red" for val in shap_values_for_input])
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.title("Feature Contributions to Prediction")
    plt.tight_layout()

    return explanation_text, plt

def main():
    # Initialize database
    create_database()

    # Front-end elements
    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f4f6f9;
            border: 2px solid #e6e8eb;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            background-color: #4caf50;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: white;
        }
        </style>
        <div class="main-container">
        <div class="header">
        <h1>Loan Prediction ML App</h1>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # User inputs
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Education_1 = st.selectbox('Education', ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    # Prediction
    if st.button("Predict"):
        result, input_data = prediction(
            Credit_History,  # Correct order
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Loan_Amount_Term
        )

        # Insert into database
        insert_into_database(Credit_History, Education_1, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, result)

        # Show result in color
        if result == "Approved":
            st.success(f'Your loan is {result}', icon="✅")
        else:
            st.error(f'Your loan is {result}', icon="❌")

        # Explanation: Feature Contributions and SHAP Bar Plot
        st.header("Explanation of Prediction")
        explanation_text, bar_chart = explain_prediction(input_data, final_result=result)
        st.write(explanation_text)
        st.pyplot(bar_chart)

if __name__ == '__main__':
    main()
