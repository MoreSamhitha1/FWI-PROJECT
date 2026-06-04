from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['DC']),
            float(request.form['ISI']),
            float(request.form['BUI']),
            float(request.form['Region'])
        ]
        input_data = np.array([data])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction = round(prediction, 2)
        if prediction <= 5:
            risk_class = "low"
            risk_label = "Low"
            advisory = "Fire risk is low. Normal activities can continue safely."  
        elif prediction <= 10:
            risk_class = "moderate"
            risk_label = "Moderate"
            advisory = "Moderate fire risk. Be cautious with fire-related activities."
        elif prediction <= 20:
            risk_class = "high"
            risk_label = "High"
            advisory = "High fire risk. Avoid open flames and monitor conditions closely."
        else:
            risk_class = "extreme"
            risk_label = "Extreme"
            advisory = "Extreme fire risk! Avoid all fire activities and follow safety protocols."
        return render_template(
            "home.html",
            prediction=prediction,
            risk_class=risk_class,
            risk_label=risk_label,
            advisory=advisory
        )
    except Exception as e:
        return f"Error: {e}"
if __name__ == "__main__":
    app.run(debug=True)
