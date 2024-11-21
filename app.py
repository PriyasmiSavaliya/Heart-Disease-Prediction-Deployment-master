from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model and scaler
model_filename = 'heart-disease-prediction-rf-model.pkl'
scaler_filename = 'scaler.pkl'

model = pickle.load(open(model_filename, 'rb'))
scaler = pickle.load(open(scaler_filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')  # The form where users input their data

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form values and convert to appropriate data types
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # Male = 1, Female = 0
        cp = int(request.form['cp'])  # Chest Pain Type (0-3)
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])  # 1 for Fasting blood sugar > 120, 0 otherwise
        restecg = int(request.form['restecg'])  # Resting ECG (0-2)
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])  # Exercise-induced Angina (1=Yes, 0=No)
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])  # Slope of peak exercise ST segment (0-2)
        ca = int(request.form['ca'])  # Number of major vessels (0-4)
        thal = int(request.form['thal'])  # Thalassemia (0-2)

        # Prepare the data for prediction
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the data using the loaded scaler
        data_scaled = scaler.transform(data)

        # Make the prediction
        prediction = model.predict(data_scaled)

        print(prediction)
        # Interpret the prediction result
        # if prediction[0] == 1:
        #     result = "Heart Disease"
        # else:
        #     result = "No Heart Disease"


        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
