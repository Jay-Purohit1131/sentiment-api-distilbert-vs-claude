import os
from unittest.mock import MagicMock, patch
import torch


def _fake_model_output():
    """A fake model call returning logits that softmax to a clear 'positive'."""
    out = MagicMock()
    out.logits = torch.tensor([[0.1, 5.0]])   # index 1 (positive) dominates
    return out


# Patch the HF loaders so importing predict.py / app.py never touches disk or the Hub.
_tokenizer = MagicMock()
_tokenizer.return_value = {"input_ids": torch.tensor([[101, 102]]),
                           "attention_mask": torch.tensor([[1, 1]])}

_model = MagicMock()
_model.return_value = _fake_model_output()
_model.config.id2label = {0: "negative", 1: "positive"}

patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer).start()
patch("transformers.AutoModelForSequenceClassification.from_pretrained",
      return_value=_model).start()


# Dummy key so `Anthropic()` can be constructed at import time.
# Real calls are mocked in tests, so this key is never actually used.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")