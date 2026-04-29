from flask import Flask, render_template, request
import numpy as np
import pickle
from datetime import datetime

# ---------- Visualization ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# ===============================
# LOAD MODEL, SCALER & ENCODERS
# ===============================
with open("smr_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
gender_encoder = saved["gender_encoder"]
medication_encoder = saved["medication_encoder"]
condition_encoder = saved["condition_encoder"]

MEDICATIONS = list(medication_encoder.classes_)
CONDITIONS = list(condition_encoder.classes_)
GENDERS = ["Male", "Female"]

# ===============================
# HOME ROUTE
# ===============================
@app.route("/")
def home():
    return render_template(
        "index.html",
        medications=MEDICATIONS,
        conditions=CONDITIONS,
        genders=GENDERS
    )

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------- INPUTS ----------
        age = int(request.form["age"])
        dosage = float(request.form["dosage"])
        reminder_sent = int(request.form["reminder_sent"])

        gender = "M" if request.form["gender"] == "Male" else "F"
        gender_enc = gender_encoder.transform([gender])[0]

        medication = request.form["medication"]
        medication_enc = medication_encoder.transform([medication])[0]

        condition = request.form["condition"]
        condition_enc = condition_encoder.transform([condition])[0]

        # ---------- FEATURE VECTOR ----------
        input_data = np.array([[
            age,
            gender_enc,
            medication_enc,
            condition_enc,
            dosage,
            reminder_sent
        ]])

        # ---------- SCALE ----------
        input_scaled = scaler.transform(input_data)

        # Model predicts: 1 = WILL MISS medication
        prob_miss = model.predict_proba(input_scaled)[0][1]
        prob_taken = 1 - prob_miss

        # ---------- RISK LOGIC ----------
        if prob_taken < 0.45:
            alert_type = "danger"
            alert_msg = "🚨 HIGH RISK: Medication likely to be missed!"
            reminder_msg = "⏰ Immediate reminder sent to patient."
        elif prob_taken < 0.7:
            alert_type = "warning"
            alert_msg = "⚠️ MODERATE RISK: Patient may forget medication."
            reminder_msg = "🔔 Reminder scheduled."
        else:
            alert_type = "success"
            alert_msg = "✅ LOW RISK: Patient is adherent."
            reminder_msg = "👍 No reminder required."

        # ---------- VISUALIZATION ----------
        fig = plt.figure(figsize=(4, 3))
        plt.bar(
            ["Will Take", "Will Miss"],
            [prob_taken, prob_miss]
        )
        plt.ylim(0, 1)
        plt.title("Medication Adherence Probability")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        risk_chart = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return render_template(
            "index.html",
            medications=MEDICATIONS,
            conditions=CONDITIONS,
            genders=GENDERS,
            alert_type=alert_type,
            alert_msg=alert_msg,
            reminder_msg=reminder_msg,
            confidence=f"{prob_taken*100:.1f}%",
            time_now=datetime.now().strftime("%d-%m-%Y %H:%M"),
            risk_chart=risk_chart
        )

    except Exception as e:
        return render_template(
            "index.html",
            medications=MEDICATIONS,
            conditions=CONDITIONS,
            genders=GENDERS,
            alert_type="danger",
            alert_msg=f"Error occurred: {str(e)}"
        )

# ===============================
# RUN APPLICATION
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
