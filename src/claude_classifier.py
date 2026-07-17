"""
Claude-based sentiment classifier - the LLM counterpart to predict.py
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = 'claude-haiku-4-5'

SYSTEM_PROMPT = (
    "You classify the overall sentiment of a movie review as either positive or "
    "negative. Judge the reviewer's overall verdict on the film, not individual "
    "words: sarcastic praise counts as negative, and a review that ends on a "
    "strong negative note is negative even if it opens positively. "
    "Respond with exactly one word, lowercase: either 'positive' or 'negative'. "
    "No punctuation, no explanation."
)


def classify(text: str) -> dict:
    """
    Classify one review. Returns {label, condifence, input tokens, output tokens}
    """
    msg = client.messages.create(
        model = MODEL,
        max_tokens = 5,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = msg.content[0].text.strip().lower()

    if "positive" in raw:
        label = "positive"
    elif "negative" in raw:
        label = "negative"
    else:
        label = "unknown"                 # Flagging anything unexpected rather than guessing

    return{
        "label": label,
        "confidence" : None,
        "input_tokens" : msg.usage.input_tokens,
        "output_tokens" : msg.usage.output_tokens,
    }

if __name__ == "__main__":
    samples = [
        "An absolute masterpiece from start to finish.",
        "Oh brilliant, another two hours I'll never get back. Just wonderful.",
        "The cinematography is stunning, but the script is lazy and the pacing drags.",
    ]
    
    for s in samples:
        print(s)
        print(classify(s), "\n")
