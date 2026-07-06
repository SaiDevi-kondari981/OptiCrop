from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask application
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)

    return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)