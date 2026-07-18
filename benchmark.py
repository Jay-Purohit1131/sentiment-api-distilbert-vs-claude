# Latency benchmark: warm-up, then multiple runs, report 50-95-99 percentiles

import time
import statistics
from src.predict import predict
from src.claude_classifier import classify

SAMPLE = " A gripping, beautifully acted film that stayed with me for days."


def benchmark (fn, name, runs=50, warmup=3):
    for _ in range(warmup):
        fn(SAMPLE)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(SAMPLE)
        times.append((time.perf_counter() - t0)*1000)
    times.sort()
    print(f"{name:<22} p50 = {statistics.median(times):7.2f}ms "
          f"p95={times[int(0.95*len(times))]:7.2f}ms "
          f"p99={times[int(0.99*len(times))]:7.2f}ms "
          f"mean={statistics.mean(times):7.2f}ms"
          )
    

if __name__ == "__main__":
    benchmark(predict, "DistilBERT (local)", runs=100)
    benchmark(classify, "Claude (Haiku 4.5)", runs=30)