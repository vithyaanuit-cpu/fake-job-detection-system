import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("fake_real_job_postings.csv")

# ===== TEXT CLEANING STEP =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data['job_title'] = data['job_title'].apply(clean_text)
data['job_description'] = data['job_description'].apply(clean_text)
# ==============================

# Combine text columns
data['text'] = data['job_title'].fillna('') + " " + data['job_description'].fillna('')

X = data['text']
y = data['is_fake']

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Scores:", scores)
print("Average Score:", scores.mean())

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))