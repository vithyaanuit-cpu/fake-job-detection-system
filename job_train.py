import pandas as pd
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("job_train.csv")
print("Dataset Shape:", df.shape)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["text"] = (
    df["title"].fillna('') + " " +
    df["description"].fillna('') + " " +
    df["requirements"].fillna('')
)

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["fraudulent"]

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(X)

# Train-Test-Validation Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train size:", len(y_train))
print("Validation size:", len(y_val))
print("Test size:", len(y_test))

# Model with class balance
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Validation
val_pred = model.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, val_pred))

# Cross Validation
scores = cross_val_score(model, X, y, cv=5)
print("\nCross Validation Scores:", scores)
print("Average CV Score:", scores.mean())

# Final Test
y_pred = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))