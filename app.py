from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic.pkl')

# Updated UI layout with modern styling and responsive grid
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: #334155;
        }
        .container {
            width: 100%;
            max-width: 680px;
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(12px);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h2 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p {
            color: #64748b;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            background: #f8fafc;
            border: 1.5px solid #e2e8f0;
            border-radius: 10px;
            font-size: 14px;
            color: #1e293b;
            transition: all 0.2s ease;
            outline: none;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }
        button {
            grid-column: span 2;
            margin-top: 10px;
            padding: 14px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
            background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        }
        button:active {
            transform: translateY(0);
        }
        .result-box {
            margin-top: 25px;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-size: 16px;
            font-weight: 700;
            animation: fadeIn 0.3s ease-in-out;
        }
        .result-approved {
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
        }
        .result-rejected {
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fca5a5;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width, button {
                grid-column: span 1;
            }
            .container {
                padding: 24px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Loan Eligibility Chek..!</h2>
            <p>Enter applicant details below to evaluate approval eligibility</p>
        </div>
        <form action="/predict" method="POST" class="form-grid">
            <div class="form-group">
                <label>Gender</label>
                <select name="Gender">
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>Married Status</label>
                <select name="Married">
                    <option value="1">Married</option>
                    <option value="0">Single</option>
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
                <label>Credit History</label>
                <select name="Credit_History">
                    <option value="1">1.0 (Good Credit)</option>
                    <option value="0">0.0 (Bad Credit)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Applicant Income ($)</label>
                <input type="number" step="any" name="ApplicantIncome" placeholder="e.g. 5000" required>
            </div>
            <div class="form-group">
                <label>Coapplicant Income ($)</label>
                <input type="number" step="any" name="CoapplicantIncome" placeholder="e.g. 1500" required>
            </div>
            <div class="form-group">
                <label>Loan Amount (Thousands)</label>
                <input type="number" step="any" name="LoanAmount" placeholder="e.g. 150" required>
            </div>
            <div class="form-group">
                <label>Loan Term (Days/Months)</label>
                <input type="number" step="any" name="Loan_Amount_Term" value="360" required>
            </div>
            <div class="form-group full-width">
                <label>Property Area</label>
                <select name="Property_Area">
                    <option value="2">Urban</option>
                    <option value="1">Semiurban</option>
                    <option value="0">Rural</option>
                </select>
            </div>
            <button type="submit">Predict Eligibility</button>
        </form>

        {% if prediction_text %}
            <div class="result-box {{ result_class }}">
                {{ prediction_text }}
            </div>
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
        
        if prediction[0] == 1:
            result = "Loan Status: Approved (Eligible)"
            result_class = "result-approved"
        else:
            result = "Loan Status: Rejected (Not Eligible)"
            result_class = "result-rejected"

        return render_template_string(HTML_TEMPLATE, prediction_text=result, result_class=result_class)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}", result_class="result-rejected")

if __name__ == '__main__':
    app.run(debug=True)
