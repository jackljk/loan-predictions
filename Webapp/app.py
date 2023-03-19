from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import webbrowser
import subprocess
import sys


dict_grades = {'AA1': 1,
               'AA2': 2,
               'AA3': 3,
               'AA4': 4,
               'AA5': 5,
               'AB1': 6,
               'AB2': 7,
               'AB3': 8,
               'AB4': 9,
               'AB5': 10,
               'AC1': 11,
               'AC2': 12,
               'AC3': 13,
               'AC4': 14,
               'AC5': 15,
               'AD1': 16,
               'AD2': 17,
               'AD3': 18,
               'AD4': 19,
               'AD5': 20,
               'AE1': 21,
               'AE2': 22,
               'AE3': 23,
               'AE4': 24,
               'AE5': 25,
               'AF1': 26,
               'AF2': 27,
               'AF3': 28,
               'AF4': 29,
               'AF5': 30,
               'AG1': 31,
               'AG2': 32,
               'AG3': 33,
               'AG4': 34,
               'AG5': 35,
               'BA1': 36,
               'BA2': 37,
               'BA3': 38,
               'BA4': 39,
               'BA5': 40,
               'BB1': 41,
               'BB2': 42,
               'BB3': 43,
               'BB4': 44,
               'BB5': 45,
               'BC1': 46,
               'BC2': 47,
               'BC3': 48,
               'BC4': 49,
               'BC5': 50,
               'BD1': 51,
               'BD2': 52,
               'BD3': 53,
               'BD4': 54,
               'BD5': 55,
               'BE1': 56,
               'BE2': 57,
               'BE3': 58,
               'BE4': 59,
               'BE5': 60,
               'BF1': 61,
               'BF2': 62,
               'BF3': 63,
               'BF4': 64,
               'BF5': 65,
               'BG1': 66,
               'BG2': 67,
               'BG3': 68,
               'BG4': 69,
               'BG5': 70,
               'CA1': 71,
               'CA2': 72,
               'CA3': 73,
               'CA4': 74,
               'CA5': 75,
               'CB1': 76,
               'CB2': 77,
               'CB3': 78,
               'CB4': 79,
               'CB5': 80,
               'CC1': 81,
               'CC2': 82,
               'CC3': 83,
               'CC4': 84,
               'CC5': 85,
               'CD1': 86,
               'CD2': 87,
               'CD3': 88,
               'CD4': 89,
               'CD5': 90,
               'CE1': 91,
               'CE2': 92,
               'CE3': 93,
               'CE4': 94,
               'CE5': 95,
               'CF1': 96,
               'CF2': 97,
               'CF3': 98,
               'CF4': 99,
               'CF5': 100,
               'CG1': 101,
               'CG2': 102,
               'CG3': 103,
               'CG4': 104,
               'CG5': 105,
               'DA1': 106,
               'DA2': 107,
               'DA3': 108,
               'DA4': 109,
               'DA5': 110,
               'DB1': 111,
               'DB2': 112,
               'DB3': 113,
               'DB4': 114,
               'DB5': 115,
               'DC1': 116,
               'DC2': 117,
               'DC3': 118,
               'DC4': 119,
               'DC5': 120,
               'DD1': 121,
               'DD2': 122,
               'DD3': 123,
               'DD4': 124,
               'DD5': 125,
               'DE1': 126,
               'DE2': 127,
               'DE3': 128,
               'DE4': 129,
               'DE5': 130,
               'DF1': 131,
               'DF2': 132,
               'DF3': 133,
               'DF4': 134,
               'DF5': 135,
               'DG1': 136,
               'DG2': 137,
               'DG3': 138,
               'DG4': 139,
               'DG5': 140,
               'EA1': 141,
               'EA2': 142,
               'EA3': 143,
               'EA4': 144,
               'EA5': 145,
               'EB1': 146,
               'EB2': 147,
               'EB3': 148,
               'EB4': 149,
               'EB5': 150,
               'EC1': 151,
               'EC2': 152,
               'EC3': 153,
               'EC4': 154,
               'EC5': 155,
               'ED1': 156,
               'ED2': 157,
               'ED3': 158,
               'ED4': 159,
               'ED5': 160,
               'EE1': 161,
               'EE2': 162,
               'EE3': 163,
               'EE4': 164,
               'EE5': 165,
               'EF1': 166,
               'EF2': 167,
               'EF3': 168,
               'EF4': 169,
               'EF5': 170,
               'EG1': 171,
               'EG2': 172,
               'EG3': 173,
               'EG4': 174,
               'EG5': 175,
               'FA1': 176,
               'FA2': 177,
               'FA3': 178,
               'FA4': 179,
               'FA5': 180,
               'FB1': 181,
               'FB2': 182,
               'FB3': 183,
               'FB4': 184,
               'FB5': 185,
               'FC1': 186,
               'FC2': 187,
               'FC3': 188,
               'FC4': 189,
               'FC5': 190,
               'FD1': 191,
               'FD2': 192,
               'FD3': 193,
               'FD4': 194,
               'FD5': 195,
               'FE1': 196,
               'FE2': 197,
               'FE3': 198,
               'FE4': 199,
               'FE5': 200,
               'FF1': 201,
               'FF2': 202,
               'FF3': 203,
               'FF4': 204,
               'FF5': 205,
               'FG1': 206,
               'FG2': 207,
               'FG3': 208,
               'FG4': 209,
               'FG5': 210,
               'GA1': 211,
               'GA2': 212,
               'GA3': 213,
               'GA4': 214,
               'GA5': 215,
               'GB1': 216,
               'GB2': 217,
               'GB3': 218,
               'GB4': 219,
               'GB5': 220,
               'GC1': 221,
               'GC2': 222,
               'GC3': 223,
               'GC4': 224,
               'GC5': 225,
               'GD1': 226,
               'GD2': 227,
               'GD3': 228,
               'GD4': 229,
               'GD5': 230,
               'GE1': 231,
               'GE2': 232,
               'GE3': 233,
               'GE4': 234,
               'GE5': 235,
               'GF1': 236,
               'GF2': 237,
               'GF3': 238,
               'GF4': 239,
               'GF5': 240,
               'GG1': 241,
               'GG2': 242,
               'GG3': 243,
               'GG4': 244,
               'GG5': 245}

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('dt_model.pkl')

def open_browser():
    url = "http://localhost:5000"
    webbrowser.open_new(url)

def run_app():
    subprocess.Popen([sys.executable, "-m", "flask", "run"])
    
def show_server_banner(env, app_debug, app_port):
    if sys.stdout is not None:
        from click import echo
        echo(f"Server running on {env} mode, Debug={app_debug} in http://localhost:{app_port}")

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')


# Custom error handler for handling errors when a form is submitted
@app.errorhandler(500)
def handle_submit_error(error):
    # Log the error
    app.logger.error('An error occurred while processing the form submission: %s', error)

    # Render an error page with a custom message
    return render_template('error.html', message='An error occurred while processing your form submission.'), 500


# Define the route for the prediction result
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input
        loan_amount = request.form['loan_amount']
        funded_amount_investor = request.form['funded_amount_investor']
        interest_rate = request.form['interest_rate']
        grade = request.form['grade']
        sub_grade = request.form['subgrade']
        employment_duration = request.form['employment_duration']
        home_ownnenrship = request.form['home_ownership']
        verification_status = request.form['verification_status']
        debit_to_income = request.form['debit_to_income']
        delinquencies = request.form['delinquency_two_years']
        public_record = request.form['public_record']
        total_received_interest = request.form['total_received_interest']
        total_received_late_fee = request.form['total_received_late_fee']
        collection_12_months_medical = request.form['collection_12_months_medical']
        application_type = request.form['application_type']
        total_collection_amount = request.form['total_collection_amount']
        total_current_balance = request.form['total_current_balance']
        total_revolving_credit_limit = request.form['total_revolving_credit_limit']

        # Preprocess the input
        grades_transformed = dict_grades[grade + sub_grade]
        preproc = ColumnTransformer([
        ("one-hot", OneHotEncoder(handle_unknown="ignore"), ["Employment Duration", "Verification Status", "Application Type"]),
        ("Std-scale", StandardScaler(), ['Total Collection Amount', 'Total Received Late Fee'])
    ], remainder="passthrough")

        X = pd.DataFrame([[loan_amount, funded_amount_investor, interest_rate, grades_transformed, employment_duration, home_ownnenrship, verification_status, debit_to_income, delinquencies, public_record, total_received_interest, total_received_late_fee, collection_12_months_medical, application_type, total_collection_amount, total_current_balance, total_revolving_credit_limit]], 
                        columns=['Loan Amount', 'Funded Amount Investor', 'Interest Rate', 'GradeSubGrade', 'Employment Duration', 'Home Ownership', 'Verification Status', 'Debit to Income', 'Delinquency - two years', 'Public Record', 'Total Received Interest', 'Total Received Late Fee', 'Collection 12 months Medical', 'Application Type', 'Total Collection Amount', 'Total Current Balance', 'Total Revolving Credit Limit'])

        # Make the prediction
        prediction = model.predict(X)

        if prediction[0] == 0:
            prediction = 'Not Approved'
        else:
            prediction = 'Approved'

        # Render the prediction result template
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return render_template('error.html', message='An error occurred while processing your form submission.'), 500
    

if __name__ == '__main__':
    app.run(debug=True)
