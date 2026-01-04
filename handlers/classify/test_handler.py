"""
Tests for the classification handler.

Run with: pytest test_handler.py -v
"""

import math
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

import requests

# Module-level imports for handler functions (reduces repetition)
from handler import (
    parse_json_response,
    validate_classification_result,
    build_classification_prompt,
    fetch_image,
    classify,
    handler,
)


# =============================================================================
# Tests for parse_json_response()
# =============================================================================

class TestParseJsonResponse:
    """Tests for JSON parsing from model responses."""

    def test_parse_markdown_json_code_block(self):
        """Test parsing JSON from markdown code block with json specifier."""
        text = '```json\n{"type": "W2", "confidence": 0.95}\n```'
        result = parse_json_response(text)
        assert result["type"] == "W2"
        assert result["confidence"] == 0.95

    def test_parse_raw_json(self):
        """Test parsing raw JSON without code blocks."""
        text = '{"type": "bank_statement", "confidence": 0.87}'
        result = parse_json_response(text)
        assert result["type"] == "bank_statement"
        assert result["confidence"] == 0.87

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in extra text."""
        text = 'Here is the result:\n{"type": "invoice", "confidence": 0.9}\nDone.'
        result = parse_json_response(text)
        assert result["type"] == "invoice"

    def test_parse_code_block_without_json_specifier(self):
        """Test parsing JSON from code block without language specifier."""
        text = '```\n{"type": "receipt", "confidence": 0.85}\n```'
        result = parse_json_response(text)
        assert result["type"] == "receipt"

    def test_parse_invalid_json_returns_empty_dict(self):
        """Test that invalid JSON returns empty dict."""
        text = 'This is not JSON at all'
        result = parse_json_response(text)
        assert result == {}

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty dict."""
        result = parse_json_response('')
        assert result == {}

    def test_parse_json_with_nested_objects(self):
        """Test parsing JSON with nested structure."""
        text = '{"type": "W2", "confidence": 0.9, "metadata": {"page": 1}}'
        result = parse_json_response(text)
        assert result["type"] == "W2"
        assert result["metadata"]["page"] == 1


# =============================================================================
# Tests for validate_classification_result()
# =============================================================================

class TestValidateClassificationResult:
    """Tests for classification result validation."""

    def test_valid_result_passes_through(self):
        """Test that valid results pass through unchanged."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": 0.95,
            "reasoning": "test"
        })
        assert result["type"] == "W2"
        assert result["confidence"] == 0.95
        assert result["reasoning"] == "test"

    def test_invalid_type_defaults_to_other(self):
        """Test that invalid document types default to 'other'."""
        result = validate_classification_result({
            "type": "invalid_type",
            "confidence": 0.8
        })
        assert result["type"] == "other"

    def test_confidence_too_high_clamps_to_one(self):
        """Test that confidence > 1.0 is clamped to 1.0."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": 1.5
        })
        assert result["confidence"] == 1.0

    def test_confidence_too_low_clamps_to_zero(self):
        """Test that confidence < 0.0 is clamped to 0.0."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": -0.5
        })
        assert result["confidence"] == 0.0

    def test_non_numeric_confidence_defaults(self):
        """Test that non-numeric confidence defaults to 0.5."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": "high"
        })
        assert result["confidence"] == 0.5

    def test_missing_fields_get_defaults(self):
        """Test that missing fields receive default values."""
        result = validate_classification_result({})
        assert result["type"] == "other"
        assert result["confidence"] == 0.5
        assert result["reasoning"] == ""

    @pytest.mark.parametrize("doc_type", [
        'W2', '1099-INT', '1099-DIV', '1099-MISC', '1099-NEC',
        '1099-R', '1098', 'bank_statement', 'credit_card_statement',
        'invoice', 'receipt', 'check', 'other'
    ])
    def test_all_valid_document_types(self, doc_type):
        """Test that all valid document types are accepted."""
        result = validate_classification_result({"type": doc_type})
        assert result["type"] == doc_type

    def test_none_confidence_defaults(self):
        """Test that None confidence defaults to 0.5."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": None
        })
        assert result["confidence"] == 0.5

    def test_nan_confidence_clamps_to_one(self):
        """Test that NaN confidence clamps to 1.0 due to min/max behavior."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": float('nan')
        })
        # NaN passes float() but min(1.0, nan) returns 1.0 in Python
        # This is current behavior - if NaN should default to 0.5, handler needs updating
        assert result["confidence"] == 1.0

    def test_positive_inf_confidence_clamps_to_one(self):
        """Test that positive infinity confidence clamps to 1.0."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": float('inf')
        })
        assert result["confidence"] == 1.0

    def test_negative_inf_confidence_clamps_to_zero(self):
        """Test that negative infinity confidence clamps to 0.0."""
        result = validate_classification_result({
            "type": "W2",
            "confidence": float('-inf')
        })
        assert result["confidence"] == 0.0


# =============================================================================
# Tests for build_classification_prompt()
# =============================================================================

class TestBuildClassificationPrompt:
    """Tests for classification prompt generation."""

    def test_prompt_contains_document_types(self):
        """Test that prompt contains all document types."""
        prompt = build_classification_prompt()
        assert "W2" in prompt
        assert "1099-INT" in prompt
        assert "1099-DIV" in prompt
        assert "bank_statement" in prompt
        assert "invoice" in prompt
        assert "other" in prompt

    def test_prompt_requests_json_format(self):
        """Test that prompt requests JSON output format."""
        prompt = build_classification_prompt()
        assert "JSON" in prompt
        assert "type" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt

    def test_prompt_is_non_empty(self):
        """Test that prompt is not empty."""
        prompt = build_classification_prompt()
        assert len(prompt) > 100


# =============================================================================
# Tests for fetch_image()
# =============================================================================

class TestFetchImage:
    """Tests for image fetching from URLs."""

    def test_fetch_image_success(self, mock_successful_response):
        """Test successful image fetch."""
        from PIL import Image

        with patch('handler.requests.get', return_value=mock_successful_response):
            image = fetch_image("https://example.com/image.png")
            assert isinstance(image, Image.Image)
            assert image.mode == 'RGB'

    def test_fetch_image_http_error(self, mock_http_error_response):
        """Test fetch_image raises ValueError on HTTP error."""
        with patch('handler.requests.get', return_value=mock_http_error_response):
            with pytest.raises(ValueError, match="Failed to fetch image"):
                fetch_image("https://example.com/notfound.png")

    def test_fetch_image_timeout(self):
        """Test fetch_image raises ValueError on timeout."""
        with patch('handler.requests.get', side_effect=requests.Timeout("Connection timed out")):
            with pytest.raises(ValueError, match="Failed to fetch image"):
                fetch_image("https://example.com/slow.png", timeout=1)

    def test_fetch_image_connection_error(self):
        """Test fetch_image raises ValueError on connection error."""
        with patch('handler.requests.get', side_effect=requests.ConnectionError("Network unreachable")):
            with pytest.raises(ValueError, match="Failed to fetch image"):
                fetch_image("https://example.com/image.png")

    def test_fetch_image_invalid_image_data(self):
        """Test fetch_image raises ValueError on invalid image data."""
        mock_response = MagicMock()
        mock_response.content = b"not an image"
        mock_response.raise_for_status = MagicMock()

        with patch('handler.requests.get', return_value=mock_response):
            with pytest.raises(ValueError, match="Failed to decode image"):
                fetch_image("https://example.com/notanimage.txt")

    def test_fetch_image_uses_provided_timeout(self, valid_image_bytes):
        """Test that fetch_image uses the provided timeout value."""
        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.requests.get', return_value=mock_response) as mock_get:
            fetch_image("https://example.com/image.png", timeout=60)
            mock_get.assert_called_once_with("https://example.com/image.png", timeout=60)

    def test_fetch_image_converts_rgba_to_rgb(self, rgba_image_bytes):
        """Test that RGBA images are converted to RGB."""
        from PIL import Image

        mock_response = MagicMock()
        mock_response.content = rgba_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.requests.get', return_value=mock_response):
            image = fetch_image("https://example.com/image.png")
            assert image.mode == 'RGB'

    def test_fetch_image_converts_grayscale_to_rgb(self, grayscale_image_bytes):
        """Test that grayscale images are converted to RGB."""
        from PIL import Image

        mock_response = MagicMock()
        mock_response.content = grayscale_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.requests.get', return_value=mock_response):
            image = fetch_image("https://example.com/image.png")
            assert image.mode == 'RGB'

    def test_fetch_image_converts_palette_to_rgb(self, palette_image_bytes):
        """Test that palette (P mode) images are converted to RGB."""
        from PIL import Image

        mock_response = MagicMock()
        mock_response.content = palette_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.requests.get', return_value=mock_response):
            image = fetch_image("https://example.com/image.png")
            assert image.mode == 'RGB'


# =============================================================================
# Tests for handler()
# =============================================================================

class TestHandler:
    """Tests for the RunPod handler function."""

    def test_handler_success(self, sample_job, sample_classification_result):
        """Test successful handler execution."""
        with patch('handler.classify', return_value=sample_classification_result):
            result = handler(sample_job)

            assert result["type"] == "W2"
            assert result["confidence"] == 0.95
            assert "latency_ms" in result
            assert isinstance(result["latency_ms"], int)

    def test_handler_missing_image_urls(self):
        """Test handler returns error when image_urls is missing."""
        job = {"input": {}}
        result = handler(job)

        assert "error" in result
        assert "image_urls" in result["error"].lower()

    def test_handler_empty_image_urls(self):
        """Test handler returns error when image_urls is empty."""
        job = {"input": {"image_urls": []}}
        result = handler(job)

        assert "error" in result
        assert "image_urls" in result["error"].lower()

    def test_handler_empty_input(self):
        """Test handler handles empty input dict."""
        job = {}
        result = handler(job)

        assert "error" in result

    def test_handler_value_error(self, sample_job):
        """Test handler catches ValueError and returns error dict."""
        with patch('handler.classify', side_effect=ValueError("Test error")):
            result = handler(sample_job)

            assert "error" in result
            assert "Test error" in result["error"]

    def test_handler_general_exception(self, sample_job):
        """Test handler catches general exceptions and returns error dict."""
        with patch('handler.classify', side_effect=RuntimeError("Unexpected error")):
            result = handler(sample_job)

            assert "error" in result
            assert "Internal error" in result["error"]

    def test_handler_passes_custom_prompt(self, sample_job_with_prompt, sample_classification_result):
        """Test handler passes custom prompt to classify."""
        with patch('handler.classify', return_value=sample_classification_result) as mock_classify:
            handler(sample_job_with_prompt)

            mock_classify.assert_called_once_with(
                ["https://example.com/document.png"],
                "Custom classification prompt"
            )

    def test_handler_latency_is_positive(self, sample_job, sample_classification_result):
        """Test handler latency is a positive number."""
        with patch('handler.classify', return_value=sample_classification_result):
            result = handler(sample_job)

            assert result["latency_ms"] >= 0


# =============================================================================
# Tests for classify()
# =============================================================================

class TestClassify:
    """Tests for the classify function."""

    def test_classify_empty_urls_raises_error(self):
        """Test classify raises ValueError when image_urls is empty."""
        with patch('handler.load_model'):
            with pytest.raises(ValueError, match="No image URLs provided"):
                classify([])

    def test_classify_uses_first_url_only(self, valid_image_bytes):
        """Test classify only uses the first URL from the list."""
        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.load_model'), \
             patch('handler.requests.get', return_value=mock_response) as mock_get, \
             patch('handler.MODEL') as mock_model, \
             patch('handler.PROCESSOR') as mock_processor:

            # Setup mocks for model inference
            mock_model.device = 'cpu'
            mock_output = MagicMock()
            mock_output.__getitem__ = MagicMock(return_value=MagicMock())
            mock_model.generate.return_value = mock_output

            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 10)))
            mock_inputs.to.return_value = mock_inputs
            mock_processor.return_value = mock_inputs
            mock_processor.apply_chat_template.return_value = "template"
            mock_processor.batch_decode.return_value = ['{"type": "W2", "confidence": 0.9}']
            mock_processor.tokenizer.pad_token_id = 0

            classify(["https://example.com/page1.png", "https://example.com/page2.png"])

            # Verify only first URL was fetched
            mock_get.assert_called_once()
            assert "page1.png" in mock_get.call_args[0][0]

    def test_classify_uses_default_prompt(self, valid_image_bytes):
        """Test classify uses default prompt when none provided."""
        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.load_model'), \
             patch('handler.requests.get', return_value=mock_response), \
             patch('handler.MODEL') as mock_model, \
             patch('handler.PROCESSOR') as mock_processor:

            mock_model.device = 'cpu'
            mock_output = MagicMock()
            mock_output.__getitem__ = MagicMock(return_value=MagicMock())
            mock_model.generate.return_value = mock_output

            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 10)))
            mock_inputs.to.return_value = mock_inputs
            mock_processor.return_value = mock_inputs
            mock_processor.apply_chat_template.return_value = "template"
            mock_processor.batch_decode.return_value = ['{"type": "invoice", "confidence": 0.8}']
            mock_processor.tokenizer.pad_token_id = 0

            classify(["https://example.com/doc.png"])

            # Check apply_chat_template was called with default prompt content
            call_args = mock_processor.apply_chat_template.call_args[0][0]
            message_content = call_args[0]["content"]
            text_content = [c for c in message_content if c.get("type") == "text"][0]["text"]
            assert "W2" in text_content  # Default prompt contains W2

    def test_classify_uses_custom_prompt(self, valid_image_bytes):
        """Test classify uses custom prompt when provided."""
        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        custom_prompt = "My custom classification prompt"

        with patch('handler.load_model'), \
             patch('handler.requests.get', return_value=mock_response), \
             patch('handler.MODEL') as mock_model, \
             patch('handler.PROCESSOR') as mock_processor:

            mock_model.device = 'cpu'
            mock_output = MagicMock()
            mock_output.__getitem__ = MagicMock(return_value=MagicMock())
            mock_model.generate.return_value = mock_output

            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 10)))
            mock_inputs.to.return_value = mock_inputs
            mock_processor.return_value = mock_inputs
            mock_processor.apply_chat_template.return_value = "template"
            mock_processor.batch_decode.return_value = ['{"type": "other", "confidence": 0.7}']
            mock_processor.tokenizer.pad_token_id = 0

            classify(["https://example.com/doc.png"], prompt=custom_prompt)

            # Check apply_chat_template was called with custom prompt
            call_args = mock_processor.apply_chat_template.call_args[0][0]
            message_content = call_args[0]["content"]
            text_content = [c for c in message_content if c.get("type") == "text"][0]["text"]
            assert text_content == custom_prompt

    def test_classify_returns_validated_result(self, valid_image_bytes):
        """Test classify returns a validated result."""
        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.load_model'), \
             patch('handler.requests.get', return_value=mock_response), \
             patch('handler.MODEL') as mock_model, \
             patch('handler.PROCESSOR') as mock_processor:

            mock_model.device = 'cpu'
            mock_output = MagicMock()
            mock_output.__getitem__ = MagicMock(return_value=MagicMock())
            mock_model.generate.return_value = mock_output

            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 10)))
            mock_inputs.to.return_value = mock_inputs
            mock_processor.return_value = mock_inputs
            mock_processor.apply_chat_template.return_value = "template"
            mock_processor.batch_decode.return_value = ['{"type": "W2", "confidence": 0.95, "reasoning": "test"}']
            mock_processor.tokenizer.pad_token_id = 0

            result = classify(["https://example.com/doc.png"])

            assert "type" in result
            assert "confidence" in result
            assert "reasoning" in result
            assert result["type"] == "W2"


# =============================================================================
# Tests for load_model()
# =============================================================================

class TestLoadModel:
    """Tests for model loading."""

    def test_load_model_only_loads_once(self):
        """Test that model is only loaded once (singleton pattern)."""
        import handler

        # Reset globals
        original_model = handler.MODEL
        original_processor = handler.PROCESSOR

        try:
            handler.MODEL = "already_loaded"
            handler.PROCESSOR = "already_loaded"

            # This should return early without loading
            with patch('transformers.Qwen2VLForConditionalGeneration') as mock_qwen:
                handler.load_model()
                mock_qwen.from_pretrained.assert_not_called()

        finally:
            # Restore globals
            handler.MODEL = original_model
            handler.PROCESSOR = original_processor

    def test_load_model_happy_path(self):
        """Test that load_model correctly initializes MODEL and PROCESSOR."""
        import handler
        import sys

        # Reset globals to None to trigger loading
        original_model = handler.MODEL
        original_processor = handler.PROCESSOR

        try:
            handler.MODEL = None
            handler.PROCESSOR = None

            mock_model_instance = MagicMock()
            mock_processor_instance = MagicMock()
            mock_torch = MagicMock()
            mock_torch.float16 = "float16"

            with patch.dict(sys.modules, {'torch': mock_torch}), \
                 patch('transformers.Qwen2VLForConditionalGeneration') as mock_qwen, \
                 patch('transformers.AutoProcessor') as mock_auto_processor:

                mock_qwen.from_pretrained.return_value = mock_model_instance
                mock_auto_processor.from_pretrained.return_value = mock_processor_instance

                handler.load_model()

                # Verify from_pretrained was called with correct model name
                mock_qwen.from_pretrained.assert_called_once()
                call_args = mock_qwen.from_pretrained.call_args
                assert "Qwen/Qwen2.5-VL-7B-Instruct" in call_args[0]

                mock_auto_processor.from_pretrained.assert_called_once()
                call_args = mock_auto_processor.from_pretrained.call_args
                assert "Qwen/Qwen2.5-VL-7B-Instruct" in call_args[0]

                # Verify globals were set
                assert handler.MODEL is mock_model_instance
                assert handler.PROCESSOR is mock_processor_instance

        finally:
            # Restore globals
            handler.MODEL = original_model
            handler.PROCESSOR = original_processor


# =============================================================================
# Run tests directly (for backward compatibility)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
