import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import os

# Hugging Face model repo
HF_REPO = "jcksnpaul/movie-sentiment-analysis"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Download tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(HF_REPO, subfolder="models/final_distilbert_model", trust_remote_code=True, force_download=True)

# Download ONNX model from Hugging Face
onnx_path = hf_hub_download(
    repo_id=HF_REPO,
    filename="models/distilbert_sentiment.onnx",
    local_dir=MODELS_DIR
)

# Load ONNX model
session = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

LABELS = ["negative", "positive"]

def predict(texts):
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="np"
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    preds = np.argmax(logits, axis=1)
    return [LABELS[p] for p in preds]
