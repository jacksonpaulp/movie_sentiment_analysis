from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="Movie Sentiment API")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    sentiment = predict([input.text])[0]
    return {
        "text": input.text,
        "sentiment": sentiment
    }
