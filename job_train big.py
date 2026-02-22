# Fake Job Detection Project (Improved Version)

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "job_train.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df = df.fillna("")

# Combine important text columns
df["text"] = (
    df["title"] + " " +
    df["location"] + " " +
    df["description"] + " " +
    df["requirements"]
)

# Include additional features
X_text = df["text"]
X_features = df[["telecommuting", "has_company_logo", "has_questions"]]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_vec = vectorizer.fit_transform(X_text)

# Combine text features with other features
from scipy.sparse import hstack
X = hstack([X_text_vec, X_features.values])

y = df["fraudulent"]   # 0 = Real, 1 = Fake

# -----------------------------
# 3. Train / Validation / Test Split
# -----------------------------
# First split train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

# -----------------------------
# 4. Model Training
# -----------------------------
model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# 5. Validation Accuracy
# -----------------------------
val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)

print("\nValidation Accuracy:", val_accuracy)

# -----------------------------
# 6. Cross Validation
# -----------------------------
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5
)

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))

# -----------------------------
# 8. Test Evaluation
# -----------------------------
test_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

# -----------------------------
# 8. Predict New Job Post
# -----------------------------
def predict_job(text, telecommuting=0, has_company_logo=0, has_questions=0):
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    
    # Create feature array
    features = np.array([[telecommuting, has_company_logo, has_questions]])
    
    # Combine
    combined = hstack([text_vec, features])
    
    prediction = model.predict(combined)[0]

    if prediction == 1:
        return "Fake Job"
    else:
        return "Real Job"


# Example
example = """
Work from home. Earn $3000 per week. No experience required.
"""

print("\nPrediction Example:", predict_job(example))
 
# Save trained model and vectorizer for inference
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nSaved model to model.pkl and vectorizer to vectorizer.pkl")