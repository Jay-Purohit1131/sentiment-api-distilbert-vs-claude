from unittest.mock import patch, MagicMock
from src import claude_classifier

def _fake_response(text, in_tok=100, out_tok=4):
    """
    Fake object shaped like a real Anthropic API response.
    """

    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage.input_tokens = in_tok
    resp.usage.output_tokens = out_tok
    return resp


def test_classify_positive():
    with patch.object(
        claude_classifier.client.messages, "create",
        return_value= _fake_response("positive"),
    ) as mock_create:
        result = claude_classifier.classify("A wonderful film.")

    assert result["label"] == "positive"
    assert result["input_tokens"] == 100
    mock_create.assert_called_once()


def test_classify_negative():
    with patch.object(
        claude_classifier.client.messages, "create",
        return_value= _fake_response("negative"),
    ):
        result = claude_classifier.classify("A terrible film.")

    assert result["label"] == "negative"


def test_classify_handles_messy_output():
    with patch.object(
        claude_classifier.client.messages, "create",
        return_value= _fake_response("Positive.\n"),
    ):
        result = claude_classifier.classify("Great!")

    assert result["label"] == "positive"