import joblib
import re
import os
import numpy as np
from scipy.sparse import hstack

if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    print("model.pkl or vectorizer.pkl not found. Run: python \"job_train big.py\" to train and save them.")
    raise SystemExit(1)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

while True:
    text = input("\nEnter job description (or type exit): ")
    if text.lower() == "exit":
        break

    # Ask for optional boolean features (defaults to 0)
    def ask_flag(prompt):
        v = input(f"{prompt} [0/1] (default 0): ")
        return int(v) if v.strip() != "" else 0

    telecommuting = ask_flag("telecommuting")
    has_company_logo = ask_flag("has_company_logo")
    has_questions = ask_flag("has_questions")

    text = clean_text(text)
    text_vec = vectorizer.transform([text])

    features = np.array([[telecommuting, has_company_logo, has_questions]])
    X = hstack([text_vec, features])
    prediction = model.predict(X)

    print("Prediction:", "FAKE JOB" if prediction[0] == 1 else "REAL JOB")