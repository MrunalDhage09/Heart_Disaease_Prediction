
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# =========================
# LOAD SCALER
# =========================
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# =========================
# MODEL FILES
# =========================
MODEL_FILES = {
    "Logistic Regression": "models/logisticregression_model.pkl",
    "SVM": "models/svm_model.pkl",
    "Decision Tree": "models/decisiontree_model.pkl",
    "Random Forest": "models/randomforest_model.pkl",
    "KNN": "models/knn_model.pkl"
}

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    # Collect ALL 13 feature values (ignore name & email)
    values = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    # Convert to numpy & scale
    input_data = np.array([values])
    input_scaled = scaler.transform(input_data)

    # Get selected model
    selected_model = request.form["model"]
    model_path = MODEL_FILES[selected_model]

    # Load model
    model = pickle.load(open(model_path, "rb"))

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Result
    if prediction == 1:
        result = "⚠️ Heart Disease Detected"
    else:
        result = "✅ No Heart Disease Detected"

    return render_template(
        "result.html",
        result=result,
        model=selected_model
    )

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)
