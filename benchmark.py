"""Latency benchmark: warm-up, then multiple runs, report p50/p95/p99.

Compares four inference paths:
  - DistilBERT via the API-level predict() (PyTorch, full request path)
  - Claude via classify() (network round-trip)
  - DistilBERT PyTorch (raw model inference)
  - DistilBERT ONNX  (raw model inference, ONNX Runtime)
"""
import time
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

from src.predict import predict
from src.claude_classifier import classify

SAMPLE = "A gripping, beautifully acted film that stayed with me for days."

# ── Raw model loads for the PyTorch-vs-ONNX comparison ───────────────────────
tokenizer = AutoTokenizer.from_pretrained("./distilbert-imdb")
pt_model = AutoModelForSequenceClassification.from_pretrained("./distilbert-imdb")
pt_model.eval()
onnx_model = ORTModelForSequenceClassification.from_pretrained("./distilbert-imdb-onnx")


def pt_infer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, return_token_type_ids=False)
    with torch.no_grad():
        pt_model(**inputs)


def onnx_infer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, return_token_type_ids=False)
    onnx_model(**inputs)


def benchmark(fn, name, runs=100, warmup=5):
    for _ in range(warmup):          # exclude cold start from the measurement
        fn(SAMPLE)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(SAMPLE)
        times.append((time.perf_counter() - t0) * 1000)   # ms
    times.sort()
    p50 = statistics.median(times)
    p95 = times[int(0.95 * len(times))]
    p99 = times[int(0.99 * len(times))]
    print(f"{name:<26} p50={p50:7.2f}ms  p95={p95:7.2f}ms  "
          f"p99={p99:7.2f}ms  mean={statistics.mean(times):7.2f}ms")
    return p50


if __name__ == "__main__":
    print("=== End-to-end (API path) ===")
    benchmark(predict, "DistilBERT (predict)", runs=100)
    benchmark(classify, "Claude (Haiku 4.5)", runs=30)   # fewer: paid + slow

    print("\n=== Raw inference: PyTorch vs ONNX ===")
    pt = benchmark(pt_infer, "DistilBERT (PyTorch)", runs=100)
    ox = benchmark(onnx_infer, "DistilBERT (ONNX)", runs=100)
    print(f"\nONNX speedup: {pt / ox:.2f}x")