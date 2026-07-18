"""
Standalone sentiment inference for the fine-tuned DistilBERT model.

The model and tokenizer load once when this module is imported, so the
FastAPI service in Phase 3 can import `predict` and reuse the loaded objects
across requests rather than reloading per call.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./distilbert-imdb"

# Loaded once at import — not per prediction.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()   # turn off dropout etc. for inference


def predict(text: str) -> dict:
    """Classify one review's sentiment.

    Returns e.g. {"label": "positive", "confidence": 0.9873}.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    )
    with torch.no_grad():                       # no gradients needed at inference
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]    # logits -> probabilities
    idx = int(probs.argmax())
    return {
        "label": model.config.id2label[idx],    # "negative"/"positive", from config
        "confidence": round(float(probs[idx]), 4),
    }


if __name__ == "__main__":
    samples = [
        "Is Supergirl as good a movie as Gunn’s Superman? No. But is it a fun and worthy companion piece? Absolutely.",
        "A shame the movie isn’t anywhere as good as Milly Alcock's performance.",
        "An absolute masterpiece — gripping from start to finish.",
        "A boring, predictable mess. I want my two hours back.",
    ]
    for s in samples:
        print(s)
        print(predict(s), "\n")