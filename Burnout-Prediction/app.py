from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("stress_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("login.html")

@app.route('/form')
def form():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    sleep = float(request.form['sleep'])
    work = float(request.form['work'])
    screen = float(request.form['screen'])
    stress = float(request.form['stress'])
    activity = float(request.form['activity'])
    social = float(request.form['social'])
    caffeine = float(request.form['caffeine'])

    data = np.array([[sleep, work, screen, stress, activity, social, caffeine]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    score = prediction[0]

    if score >= 7:
        result = "⚠️ High Burnout Risk"
    elif score >= 4:
        result = "⚡ Moderate Burnout Risk"
    else:
        result = "✅ Low Burnout Risk"

    return render_template("result.html", result=result, score=round(score, 2))

if __name__ == "__main__":
    app.run(debug=True)