"""Local tests for classification handler."""

import json
from unittest.mock import patch, MagicMock


def test_parse_json_response():
    from handler import parse_json_response

    # Test markdown code block
    text = '```json\n{"type": "W2", "confidence": 0.95}\n```'
    result = parse_json_response(text)
    assert result["type"] == "W2"

    # Test raw JSON
    text = '{"type": "bank_statement", "confidence": 0.87}'
    result = parse_json_response(text)
    assert result["type"] == "bank_statement"

    # Test with extra text
    text = 'Here is the result:\n{"type": "invoice", "confidence": 0.9}\nDone.'
    result = parse_json_response(text)
    assert result["type"] == "invoice"

    # Test code block without json specifier
    text = '```\n{"type": "receipt", "confidence": 0.85}\n```'
    result = parse_json_response(text)
    assert result["type"] == "receipt"

    # Test invalid JSON returns empty dict
    text = 'This is not JSON at all'
    result = parse_json_response(text)
    assert result == {}


def test_validate_classification_result():
    from handler import validate_classification_result

    # Valid result
    result = validate_classification_result({
        "type": "W2",
        "confidence": 0.95,
        "reasoning": "test"
    })
    assert result["type"] == "W2"
    assert result["confidence"] == 0.95
    assert result["reasoning"] == "test"

    # Invalid type defaults to 'other'
    result = validate_classification_result({
        "type": "invalid_type",
        "confidence": 0.8
    })
    assert result["type"] == "other"

    # Out of range confidence (too high) clamps to 1.0
    result = validate_classification_result({
        "type": "W2",
        "confidence": 1.5
    })
    assert result["confidence"] == 1.0

    # Out of range confidence (too low) clamps to 0.0
    result = validate_classification_result({
        "type": "W2",
        "confidence": -0.5
    })
    assert result["confidence"] == 0.0

    # Non-numeric confidence defaults to 0.5
    result = validate_classification_result({
        "type": "W2",
        "confidence": "high"
    })
    assert result["confidence"] == 0.5

    # Missing fields get defaults
    result = validate_classification_result({})
    assert result["type"] == "other"
    assert result["confidence"] == 0.5
    assert result["reasoning"] == ""

    # All valid document types
    valid_types = [
        'W2', '1099-INT', '1099-DIV', '1099-MISC', '1099-NEC',
        '1099-R', '1098', 'bank_statement', 'credit_card_statement',
        'invoice', 'receipt', 'check', 'other'
    ]
    for doc_type in valid_types:
        result = validate_classification_result({"type": doc_type})
        assert result["type"] == doc_type


def test_build_classification_prompt():
    from handler import build_classification_prompt

    prompt = build_classification_prompt()

    # Check prompt contains key document types
    assert "W2" in prompt
    assert "1099-INT" in prompt
    assert "bank_statement" in prompt
    assert "JSON" in prompt


if __name__ == "__main__":
    test_parse_json_response()
    print("test_parse_json_response passed")

    test_validate_classification_result()
    print("test_validate_classification_result passed")

    test_build_classification_prompt()
    print("test_build_classification_prompt passed")

    print("\nAll tests passed!")
