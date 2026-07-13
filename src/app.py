import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict          # the function, from your module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "INEED2PPP/sentiment-distilbert-imdb"
# no pipeline, no classifier — predict.py already loads the model once

app = FastAPI(
    title= "Sentiment Analysis API - DistilBERT vs Claude",
    description="Sentiment classification via a DistilBERT model fine-tuned on IMDB "
                "(test F1 0.937). Part of a fine-tune-vs-LLM comparison.",
    version="0.1.0",
)


class PredictionRequest (BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description= "The review text to classify.",
        examples=["An absolute masterpiece from start to finish"],
        )

class PredictionResponse(BaseModel):
    label: str
    confidence : float

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32,
                             description="The list of review texts to classify.")


@app.get("/health")
def health():
    return{'status':'ok', 'model':MODEL_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        result = predict(request.text)          # your function; note .text (attribute, not [])
        return PredictionResponse(**result)     # dict keys already match label/confidence
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    

@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchPredictionRequest):
    try:
        return [PredictionResponse(**predict(t)) for t in request.texts]
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
