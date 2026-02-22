# Fake Job Detection System

This project uses machine learning to detect fake job postings based on text analysis.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the model training and evaluation:
```bash
python model.py
```

The script will output the accuracy of the model on the test set.

## Dataset

The dataset `fake_real_job_postings.csv` contains job postings with features like job title, description, and a label indicating if the job is fake (`is_fake` column).

## Model

The model uses TF-IDF vectorization on combined title and description text, then trains a Logistic Regression classifier.

## Troubleshooting

- Ensure the CSV file is in the same directory as `model.py`
- If you get import errors, make sure all dependencies are installed
- The model achieves high accuracy on this dataset, but may need tuning for real-world use