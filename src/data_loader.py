import os
import joblib
import re

def find_data_dir(start=None):
    p = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(p, "data")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(p)
        if parent == p:
            raise FileNotFoundError("Could not find a 'data' directory in any parent folders")
        p = parent

def clean_html(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.lower()


def load_data():
    data_dir = find_data_dir()
    processed_path = os.path.join(data_dir, "processed")
    return (
        joblib.load(os.path.join(processed_path, "X_train.pkl")),
        joblib.load(os.path.join(processed_path, "X_test.pkl")),
        joblib.load(os.path.join(processed_path, "y_train.pkl")),
        joblib.load(os.path.join(processed_path, "y_test.pkl"))
    )
