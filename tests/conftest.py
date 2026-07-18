import os

# Dummy key so `Anthropic()` can be constructed at import time.
# Real calls are mocked in tests, so this key is never actually used.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")