# MedWise – Smart Medication Reminder & Health Analytics

Premium blue-and-white Flask dashboard built on the original repository, retaining the existing trained model, CSV and notebook.

## Windows setup
1. Extract this ZIP. Open the **Smart_Medication_Upgraded** folder in VS Code.
2. Open Terminal > New Terminal.
3. Run `python -m venv venv`.
4. Run `venv\Scripts\activate` (PowerShell).
5. Run `python -m pip install -r requirements.txt`.
6. Run `python app.py`.
7. Visit http://127.0.0.1:5000 and register a demo account.

If model loading fails due to scikit-learn version mismatch, install the version used to train the model or retrain/export it using your notebook. Never load pickle files from untrusted sources.

## Features
- Account registration and login (local SQLite)
- Patient management
- Medication scheduling
- In-browser reminders **only while the page is open** (not SMS, email or background notifications)
- ML non-adherence risk prediction using the original model
- Interactive risk chart and prediction history
- Responsive dashboard

## Limitations
Educational demo only. No real patient data; no production security, role-based permissions, CSRF protection or clinical validation. Do not deploy publicly without a security review. Risk scores depend on the original model and are not clinically validated. Dosage units follow the source dataset. The default development secret must be replaced via `FLASK_SECRET_KEY` for any deployment.
