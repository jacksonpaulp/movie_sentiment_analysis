import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/final_distilbert_model")

# Load ONNX model
session = ort.InferenceSession(
    "models/distilbert_sentiment.onnx",
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
