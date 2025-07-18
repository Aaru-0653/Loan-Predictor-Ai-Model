from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('loan_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

# Define the exact column order used in training
column_order = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    gender = int(request.form['Gender'])
    married = int(request.form['Married'])
    dependents = request.form['Dependents']
    if dependents == '3+':
        dependents = 3
    else:
        dependents = int(dependents)

    education = int(request.form['Education'])
    self_employed = int(request.form['Self_Employed'])
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    loan_amount_term = float(request.form['Loan_Amount_Term'])
    credit_history = int(request.form['Credit_History'])
    property_area = int(request.form['Property_Area'])

    # Prepare the data dictionary
    form_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([form_data])[column_order]

    # Prediction
    prediction = model.predict(input_df)
    result = 'Approved' if prediction[0] == 1 else 'Not Approved'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    # Optional: Test with known good data
    test_input = pd.DataFrame([{
        'Gender': 1,
        'Married': 1,
        'Dependents': 0,
        'Education': 1,
        'Self_Employed': 0,
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 0,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 2
    }])[column_order]
    print("🔍 Test Prediction:", model.predict(test_input))

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
