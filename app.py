from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model with its original filename
model = joblib.load('logistic.pkl')

# HTML template embedded directly into app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Predictor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #eef2f3; margin: 0; padding: 40px; }
        .container { max-width: 550px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: auto; }
        h2 { text-align: center; color: #333; margin-bottom: 25px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 6px; font-weight: 600; color: #444; font-size: 14px; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; font-size: 14px; }
        input:focus, select:focus { border-color: #007BFF; outline: none; }
        button { width: 100%; background: #007BFF; color: white; padding: 12px; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 10px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 12px; text-align: center; font-size: 16px; font-weight: bold; background: #e8f5e9; color: #2e7d32; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Loan Eligibility Checker</h2>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label>Gender</label>
                <select name="Gender">
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>Married</label>
                <select name="Married">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Dependents</label>
                <input type="number" name="Dependents" value="0" min="0" required>
            </div>
            <div class="form-group">
                <label>Education</label>
                <select name="Education">
                    <option value="1">Graduate</option>
                    <option value="0">Not Graduate</option>
                </select>
            </div>
            <div class="form-group">
                <label>Self Employed</label>
                <select name="Self_Employed">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Applicant Income</label>
                <input type="number" step="any" name="ApplicantIncome" placeholder="e.g. 5000" required>
            </div>
            <div class="form-group">
                <label>Coapplicant Income</label>
                <input type="number" step="any" name="CoapplicantIncome" placeholder="e.g. 2000" required>
            </div>
            <div class="form-group">
                <label>Loan Amount</label>
                <input type="number" step="any" name="LoanAmount" placeholder="e.g. 150" required>
            </div>
            <div class="form-group">
                <label>Loan Amount Term (Months)</label>
                <input type="number" step="any" name="Loan_Amount_Term" value="360" required>
            </div>
            <div class="form-group">
                <label>Credit History</label>
                <select name="Credit_History">
                    <option value="1">1.0 (Good)</option>
                    <option value="0">0.0 (Bad)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Property Area</label>
                <select name="Property_Area">
                    <option value="2">Urban</option>
                    <option value="1">Semiurban</option>
                    <option value="0">Rural</option>
                </select>
            </div>
            <button type="submit">Predict Status</button>
        </form>
        {% if prediction_text %}
            <div class="result">{{ prediction_text }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['Gender'])
        married = int(request.form['Married'])
        dependents = int(request.form['Dependents'])
        education = int(request.form['Education'])
        self_employed = int(request.form['Self_Employed'])
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = int(request.form['Property_Area'])

        features = [[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        ]]

        prediction = model.predict(features)
        result = 'Approved (Eligible)' if prediction[0] == 1 else 'Rejected (Not Eligible)'

        return render_template_string(HTML_TEMPLATE, prediction_text=f'Loan Status: {result}')
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
