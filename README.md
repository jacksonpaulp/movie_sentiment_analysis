# Movie Sentiment Analysis

A comprehensive sentiment analysis project on IMDB reviews using multiple machine learning and deep learning approaches. This project compares traditional ML models with state-of-the-art transformer models for binary sentiment classification (positive/negative).

**Live API:** The trained DistilBERT model is deployed as a FastAPI service and can be containerized with Docker for easy deployment.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Comparison & Findings](#model-comparison--findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running with Docker](#running-with-docker)
- [Testing the API](#testing-the-api)
- [Model Details](#model-details)

---

## Project Overview

This project explores sentiment analysis on IMDB movie reviews using both traditional machine learning and deep learning approaches:

1. **Traditional ML Models:** Logistic Regression, Random Forest, XGBoost with TF-IDF vectorization
2. **Deep Learning Models:** CNN, LSTM with word embeddings
3. **Transformer Models:** DistilBERT (Fine-tuned - Production Model)

The final production model is **DistilBERT**, which provides the best accuracy and is deployed as a containerized FastAPI service.

---

## Dataset

- **Source:** IMDB 50K Movie Reviews Dataset
- **Size:** 50,000 reviews
- **Classes:** Binary (Positive: 1, Negative: 0)
- **Train/Test Split:** 80/20
- **Preprocessing:** Text cleaning, tokenization, padding (max_length=256)

### Key Findings from EDA:
- Dataset is well-balanced with equal positive and negative reviews
- Average review length: ~250 words
- Common positive keywords: excellent, great, loved, amazing
- Common negative keywords: terrible, waste, awful, boring

---

## Model Comparison & Findings

### 1. **Logistic Regression + TF-IDF** ✓
- **Accuracy:** ~90.06%
- **Pros:** Fast, interpretable, good baseline
- **Cons:** Limited context understanding, struggles with sarcasm
- **Use Case:** Lightweight, resource-constrained environments

### 2. **Random Forest + TF-IDF**
- **Accuracy:** ~84.51%
- **Pros:** Non-linear decision boundaries, feature importance
- **Cons:** Slower inference, memory intensive, performed worse than logistic regression

### 3. **XGBoost + TF-IDF**
- **Accuracy:** ~86.35%
- **Pros:** Strong performance, gradient boosting advantages, performed better than Random forest
- **Cons:** Hyperparameter tuning required, slower than LR and random forest, performs worse than logistic regression for text data

### 4. **CNN (Convolutional Neural Network)**
- **Accuracy:** ~88.08%
- **Architecture:** Embedding → Conv1D → GlobalMaxPooling → Dense
- **Pros:** Good at capturing local patterns (n-grams), performed better than the tree models 
- **Cons:** Limited long-range dependencies
- **Training Time:** ~2 minutes (GPU)

### 5. **LSTM (Long Short-Term Memory)**
- **Accuracy:** ~86.73%
- **Architecture:** Embedding → LSTM → Dense layers
- **Pros:** Captures sequential dependencies, better context
- **Cons:** Slower than CNN, more parameters, real world performance on IMDB data was slightly worse
- **Training Time:** ~5 minutes (GPU)

### 6. **DistilBERT (Fine-tuned)** ⭐ **PRODUCTION MODEL**
- **Accuracy:** ~92.44%
- **Architecture:** Transformer-based (6 layers, 66M parameters)
- **Pros:** 
  - Best performance (94% accuracy)
  - Pre-trained on large corpus (bidirectional context)
  - Handles sarcasm and complex language
- **Cons:** Larger model size (~250MB), requires GPU for training
- **Training Time:** ~15 minutes (single GPU)
- **Model Size:** ~250MB (safetensors format)
- **Inference Speed:** ~50ms per review

### Performance Summary Table

| Model | Accuracy | Speed |
|-------|----------|----------|
| Logistic Regression | 90% | ⚡⚡⚡ |
| Random Forest | 85% | ⚡⚡ |
| XGBoost | 86% | ⚡⚡ |
| CNN | 88% | ⚡⚡ |
| LSTM | 87% | ⚡ |
| **DistilBERT** | **92%** | **⚡⚡** |

---

## Project Structure

```
movie_sentiment_analysis/
├── notebooks/
│   ├── eda.ipynb                          # Exploratory Data Analysis and clasic ML models
│   ├── dl_models.ipynb                    # CNN & LSTM implementations
│   ├── transformermodels.ipynb            # DistilBERT fine-tuning
│   └── inference_notebook.ipynb           # Model inference examples
├── src/
│   ├── data_loader.py                     # Data loading utilities
│   ├── dataset.py                         # PyTorch Dataset classes
│   └── __pycache__/
├── models/
│   ├── final_distilbert_model/            # Production DistilBERT model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── distilbert_sentiment.onnx          # ONNX-optimized model
│   ├── logistic_regression.pkl
│   ├── tfidf.pkl
│   └── ...
├── data/
│   ├── raw/                               # Original IMDB dataset
│   └── processed/                         # Processed train/test splits
├── app.py                                 # FastAPI application
├── inference.py                           # Model inference logic
├── Dockerfile                             # Docker configuration
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## Installation

### Prerequisites
- Python 3.10
- CUDA 12.1 (for GPU support)
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jacksonpaulp/movie_sentiment_analysis.git
   cd movie_sentiment_analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files:**
   Models are automatically downloaded from Hugging Face Hub when you run the application for the first time.

---

## Running with Docker

### Build and Run the Docker Image

1. **Build the Docker image:**
   ```bash
   docker build -t movie-sentiment-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 movie-sentiment-api
   ```

   The API will be available at `http://localhost:8000`

### Docker Configuration Details

- **Base Image:** Python 3.10-slim
- **Port:** 8000
- **Models:** Downloaded automatically from Hugging Face Hub on first run
- **Framework:** FastAPI with Uvicorn

The Dockerfile automatically downloads the DistilBERT model and ONNX files from the Hugging Face Hub (`jcksnpaul/movie-sentiment-analysis`), so no local model files need to be included.

---

## Testing the API

### Method 1: FastAPI Interactive Docs

1. **Start the API:**
   ```bash
   python app.py
   # OR with uvicorn
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:8000/docs
   ```

3. **Test the endpoint:**
   - Click on the POST `/predict` endpoint
   - Click "Try it out"
   - Enter your review text in the request body:
     ```json
     {
       "text": "This movie was absolutely amazing! Best film I've ever seen."
     }
     ```
   - Click "Execute"
   - See the prediction (positive or negative) with confidence scores

### Method 2: Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I loved this movie! Absolutely fantastic."}'
```

### Example Responses

**Positive Review:**
```json
{
  "text": "This movie was amazing!",
  "sentiment": "positive"
}
```

**Negative Review:**
```json
{
  "text": "Worst movie ever made.",
  "sentiment": "negative"
}
```

---

## Model Details

### DistilBERT Architecture

- **Base Model:** distilbert-base-uncased
- **Fine-tuning Epochs:** 2
- **Learning Rate:** 2e-5
- **Batch Size:** 16 (train), 32 (eval)
- **Max Sequence Length:** 256 tokens
- **Optimizer:** AdamW
- **Loss Function:** Cross-Entropy

### Model Files

**Hugging Face Hub Repository:**
- Repository ID: `jcksnpaul/movie-sentiment-analysis`
- Models are automatically downloaded on first API call
- Cached locally in the `models/` directory

**Available Formats:**
1. **PyTorch** (`model.safetensors`) - Used for fine-tuning and inference in Python
2. **ONNX** (`distilbert_sentiment.onnx`) - Optimized for production inference

### Making Predictions

The API accepts reviews of any length (up to 256 tokens will be used). The DistilBERT model:
- Tokenizes the input text
- Pads/truncates to 256 tokens
- Passes through the transformer
- Returns logits for binary classification
- Applies softmax for confidence scores

---

## Model Artifacts Storage

Models are hosted on **Hugging Face Hub** for easy distribution:
- No large files in GitHub (Git friendly)
- Automatic downloads on first use
- Easy version control and updates
- Accessible from anywhere with internet connection

To manually download models:
```python
from huggingface_hub import hf_hub_download

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained("jcksnpaul/movie-sentiment-analysis")

# Download ONNX model
onnx_path = hf_hub_download(
    repo_id="jcksnpaul/movie-sentiment-analysis",
    filename="models/distilbert_sentiment.onnx"
)
```

---

## Next Steps / Future Improvements

- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Implement model monitoring and drift detection

---

## References

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## License

This project is part of the ML Zoomcamp course project.
