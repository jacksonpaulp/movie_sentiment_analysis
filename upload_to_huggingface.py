#!/usr/bin/env python3
"""
Upload model artifacts to Hugging Face Hub
"""
from pathlib import Path
from huggingface_hub import HfApi, login
import os

# Configuration
HF_USERNAME = "jcksnpaul"
REPO_NAME = "movie-sentiment-analysis"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Paths
project_root = Path(__file__).parent
models_dir = project_root / "models"

def upload_models():
    """Upload models to Hugging Face Hub"""
    
    # First, login to Hugging Face
    print("Please provide your Hugging Face token when prompted...")
    print("Get it from: https://huggingface.co/settings/tokens")
    login()
    
    # Create API client
    api = HfApi()
    
    # Create repo if it doesn't exist
    print(f"\nCreating/accessing repo: {REPO_ID}")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload model files
    files_to_upload = [
        "models/final_distilbert_model/model.safetensors",
        "models/final_distilbert_model/config.json",
        "models/final_distilbert_model/tokenizer.json",
        "models/final_distilbert_model/tokenizer_config.json",
        "models/final_distilbert_model/special_tokens_map.json",
        "models/final_distilbert_model/vocab.txt",
        "models/distilbert_sentiment.onnx",
    ]
    
    for file_path in files_to_upload:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"Uploading {file_path}...")
            api.upload_file(
                path_or_fileobj=str(full_path),
                path_in_repo=file_path,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"✓ Uploaded {file_path}")
        else:
            print(f"✗ File not found: {full_path}")
    
    print(f"\n✓ Done! Models uploaded to: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_models()
