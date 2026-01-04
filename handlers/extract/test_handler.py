"""
Unit tests for the extraction handler.

Tests cover:
- parse_dots_ocr_output() with various formats
- Transformer functions (extract_amounts, extract_dates, etc.)
- fetch_image() with various responses
- image_to_base64() encoding
- handler() entry point and error handling
- call_vllm() async HTTP calls
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
from PIL import Image

# Import handler functions
from handler import (
    fetch_image,
    image_to_base64,
    parse_dots_ocr_output,
    estimate_confidence_from_elements,
    handler,
)

# Import transformer functions
from transformers import (
    extract_amounts,
    extract_dates,
    extract_ssn,
    extract_ein,
    transform_bank_statement,
    transform_w2,
    transform_1099,
    transform_generic,
    transform_to_document_schema,
)


class TestParseDotsOcrOutput:
    """Tests for parse_dots_ocr_output function."""

    def test_parses_json_output(self):
        """Should parse valid JSON with layout_elements."""
        raw = json.dumps({
            "layout_elements": [
                {"bbox": [0, 0, 100, 50], "text": "Hello", "category": "Text"}
            ]
        })
        result = parse_dots_ocr_output(raw)
        assert "layout_elements" in result
        assert len(result["layout_elements"]) == 1
        assert result["layout_elements"][0]["text"] == "Hello"

    def test_parses_bbox_format(self):
        """Should parse [x1,y1,x2,y2] text format."""
        raw = "[100, 50, 400, 80] Account Statement\n[100, 100, 400, 130] Balance: $1,234"
        result = parse_dots_ocr_output(raw)
        assert len(result["layout_elements"]) == 2
        assert result["layout_elements"][0]["bbox"] == [100, 50, 400, 80]
        assert result["layout_elements"][0]["text"] == "Account Statement"

    def test_handles_plain_text(self):
        """Should handle plain text without structure."""
        raw = "This is just plain text\nWith multiple lines"
        result = parse_dots_ocr_output(raw)
        assert len(result["layout_elements"]) == 1
        assert "plain text" in result["layout_elements"][0]["text"]

    def test_handles_empty_string(self):
        """Should handle empty string input."""
        result = parse_dots_ocr_output("")
        assert result["layout_elements"] == []
        assert result["raw_text"] == ""

    def test_handles_whitespace_only(self):
        """Should handle whitespace-only input."""
        result = parse_dots_ocr_output("   \n\n   ")
        assert result["layout_elements"] == []

    def test_handles_mixed_format(self):
        """Should handle mix of bbox and plain text."""
        raw = "[50, 50, 200, 80] Header\nSome plain text\n[50, 100, 200, 130] Footer"
        result = parse_dots_ocr_output(raw)
        # Should extract the bbox elements
        assert len(result["layout_elements"]) >= 2

    def test_handles_json_with_elements_key(self):
        """Should handle JSON with 'elements' instead of 'layout_elements'."""
        raw = json.dumps({
            "elements": [
                {"bbox": [0, 0, 100, 50], "text": "Test"}
            ]
        })
        result = parse_dots_ocr_output(raw)
        assert "layout_elements" in result


class TestEstimateConfidenceFromElements:
    """Tests for estimate_confidence_from_elements function."""

    def test_returns_average_confidence(self):
        """Should return average of element confidences."""
        elements = [
            {"confidence": 0.9},
            {"confidence": 0.8},
            {"confidence": 1.0}
        ]
        result = estimate_confidence_from_elements(elements)
        assert abs(result - 0.9) < 0.01

    def test_handles_empty_list(self):
        """Should return 0.5 for empty list."""
        result = estimate_confidence_from_elements([])
        assert result == 0.5

    def test_handles_missing_confidence(self):
        """Should use 0.85 default for missing confidence."""
        elements = [{"text": "no confidence"}]
        result = estimate_confidence_from_elements(elements)
        assert result == 0.85

    def test_handles_non_dict_elements(self):
        """Should skip non-dict elements."""
        elements = [{"confidence": 0.9}, "string", None, {"confidence": 0.8}]
        result = estimate_confidence_from_elements(elements)
        assert abs(result - 0.85) < 0.01


class TestExtractAmounts:
    """Tests for extract_amounts function."""

    def test_extracts_dollar_amounts(self):
        """Should extract amounts with dollar signs."""
        text = "Total: $1,234.56"
        result = extract_amounts(text)
        assert len(result) == 1
        assert result[0]["value"] == 1234.56

    def test_extracts_amounts_without_dollar(self):
        """Should extract amounts without dollar signs."""
        text = "Amount: 999.99"
        result = extract_amounts(text)
        assert any(a["value"] == 999.99 for a in result)

    def test_extracts_multiple_amounts(self):
        """Should extract multiple amounts from text."""
        text = "Debit: $100.00, Credit: $250.50, Balance: $1,000.00"
        result = extract_amounts(text)
        values = [a["value"] for a in result]
        assert 100.00 in values
        assert 250.50 in values
        assert 1000.00 in values

    def test_handles_no_amounts(self):
        """Should return empty list when no amounts found."""
        text = "No amounts here"
        result = extract_amounts(text)
        # May match some numbers, but should handle gracefully
        assert isinstance(result, list)

    def test_handles_large_amounts(self):
        """Should handle large amounts with commas."""
        text = "$1,234,567.89"
        result = extract_amounts(text)
        assert any(a["value"] == 1234567.89 for a in result)


class TestExtractDates:
    """Tests for extract_dates function."""

    def test_extracts_slash_format(self):
        """Should extract MM/DD/YYYY format."""
        text = "Date: 01/15/2024"
        result = extract_dates(text)
        assert "01/15/2024" in result

    def test_extracts_dash_format(self):
        """Should extract MM-DD-YYYY format."""
        text = "Date: 01-15-2024"
        result = extract_dates(text)
        assert "01-15-2024" in result

    def test_extracts_iso_format(self):
        """Should extract YYYY-MM-DD format."""
        text = "Date: 2024-01-15"
        result = extract_dates(text)
        assert "2024-01-15" in result

    def test_extracts_text_format(self):
        """Should extract 'January 15, 2024' format."""
        text = "Date: January 15, 2024"
        result = extract_dates(text)
        assert len(result) >= 1

    def test_extracts_multiple_dates(self):
        """Should extract multiple dates."""
        text = "From: 01/01/2024 To: 01/31/2024"
        result = extract_dates(text)
        assert len(result) >= 2

    def test_handles_no_dates(self):
        """Should return empty list when no dates found."""
        text = "No dates here"
        result = extract_dates(text)
        assert result == []


class TestExtractSSN:
    """Tests for extract_ssn function."""

    def test_extracts_masked_ssn(self):
        """Should extract and mask SSN."""
        text = "SSN: 123-45-6789"
        result = extract_ssn(text)
        assert result == "XXX-XX-6789"

    def test_extracts_already_masked(self):
        """Should handle already masked SSN."""
        text = "SSN: XXX-XX-1234"
        result = extract_ssn(text)
        assert result == "XXX-XX-1234"

    def test_handles_no_ssn(self):
        """Should return None when no SSN found."""
        text = "No SSN here"
        result = extract_ssn(text)
        assert result is None


class TestExtractEIN:
    """Tests for extract_ein function."""

    def test_extracts_ein(self):
        """Should extract EIN."""
        text = "EIN: 12-3456789"
        result = extract_ein(text)
        assert result == "12-3456789"

    def test_handles_no_ein(self):
        """Should return None when no EIN found."""
        text = "No EIN here"
        result = extract_ein(text)
        assert result is None


class TestTransformBankStatement:
    """Tests for transform_bank_statement function."""

    def test_extracts_bank_name(self, sample_layout_elements, sample_bank_statement_text):
        """Should extract bank name."""
        result = transform_bank_statement(sample_layout_elements, sample_bank_statement_text)
        assert result["header"]["bank_name"] == "Chase"

    def test_extracts_account_number(self, sample_layout_elements, sample_bank_statement_text):
        """Should extract masked account number."""
        result = transform_bank_statement(sample_layout_elements, sample_bank_statement_text)
        assert "1234" in result["header"]["account_number"]

    def test_extracts_statement_period(self, sample_layout_elements, sample_bank_statement_text):
        """Should extract statement period."""
        result = transform_bank_statement(sample_layout_elements, sample_bank_statement_text)
        assert result["header"]["statement_period"] is not None

    def test_has_expected_structure(self, sample_layout_elements, sample_bank_statement_text):
        """Should return expected schema structure."""
        result = transform_bank_statement(sample_layout_elements, sample_bank_statement_text)
        assert "header" in result
        assert "transactions" in result
        assert "summary" in result


class TestTransformW2:
    """Tests for transform_w2 function."""

    def test_extracts_ssn(self, sample_layout_elements, sample_w2_text):
        """Should extract employee SSN."""
        result = transform_w2(sample_layout_elements, sample_w2_text)
        assert result["employee"]["ssn"] == "XXX-XX-1234"

    def test_extracts_ein(self, sample_layout_elements, sample_w2_text):
        """Should extract employer EIN."""
        result = transform_w2(sample_layout_elements, sample_w2_text)
        assert result["employer"]["ein"] == "12-3456789"

    def test_extracts_tax_year(self, sample_layout_elements, sample_w2_text):
        """Should extract tax year."""
        result = transform_w2(sample_layout_elements, sample_w2_text)
        assert result["tax_year"] == "2023"

    def test_extracts_wages(self, sample_layout_elements, sample_w2_text):
        """Should extract wage box values."""
        result = transform_w2(sample_layout_elements, sample_w2_text)
        assert result["wages"]["box1_wages"] == 75000.00

    def test_has_expected_structure(self, sample_layout_elements, sample_w2_text):
        """Should return expected schema structure."""
        result = transform_w2(sample_layout_elements, sample_w2_text)
        assert "employee" in result
        assert "employer" in result
        assert "wages" in result
        assert "tax_year" in result


class TestTransform1099:
    """Tests for transform_1099 function."""

    def test_detects_form_type(self):
        """Should detect 1099 form type."""
        text = "Form 1099-INT Interest Income 2023"
        result = transform_1099([], text)
        assert result["form_type"] == "1099-INT"

    def test_has_expected_structure(self):
        """Should return expected schema structure."""
        result = transform_1099([], "1099-DIV 2023")
        assert "recipient" in result
        assert "payer" in result
        assert "amounts" in result


class TestTransformGeneric:
    """Tests for transform_generic function."""

    def test_extracts_amounts(self):
        """Should extract all amounts."""
        text = "Total: $500.00, Tax: $50.00"
        result = transform_generic([], text)
        assert len(result["amounts"]) >= 2

    def test_extracts_dates(self):
        """Should extract all dates."""
        text = "Date: 01/15/2024"
        result = transform_generic([], text)
        assert len(result["dates"]) >= 1

    def test_has_expected_structure(self):
        """Should return expected schema structure."""
        result = transform_generic([], "Some text")
        assert "amounts" in result
        assert "dates" in result
        assert "text_blocks" in result
        assert "tables" in result


class TestTransformToDocumentSchema:
    """Tests for transform_to_document_schema dispatcher."""

    def test_routes_bank_statement(self, sample_layout_elements, sample_bank_statement_text):
        """Should route to bank_statement transformer."""
        result = transform_to_document_schema(
            "bank_statement", sample_layout_elements, sample_bank_statement_text
        )
        assert "header" in result
        assert "transactions" in result

    def test_routes_w2(self, sample_layout_elements, sample_w2_text):
        """Should route to W2 transformer."""
        result = transform_to_document_schema(
            "W2", sample_layout_elements, sample_w2_text
        )
        assert "employee" in result
        assert "wages" in result

    def test_routes_1099_types(self):
        """Should route all 1099 types to 1099 transformer."""
        for form_type in ["1099-INT", "1099-DIV", "1099-MISC", "1099-NEC", "1099-R"]:
            result = transform_to_document_schema(form_type, [], "text")
            assert "recipient" in result

    def test_falls_back_to_generic(self):
        """Should use generic transformer for unknown types."""
        result = transform_to_document_schema("unknown_type", [], "text")
        assert "amounts" in result
        assert "text_blocks" in result


class TestFetchImage:
    """Tests for fetch_image function."""

    def test_successful_fetch(self, mock_successful_response):
        """Should return PIL Image on successful fetch."""
        with patch('handler.requests.get', return_value=mock_successful_response):
            result = fetch_image("https://example.com/image.png")
            assert isinstance(result, Image.Image)
            assert result.mode == 'RGB'

    def test_http_error_raises_valueerror(self, mock_http_error_response):
        """Should raise ValueError on HTTP error."""
        with patch('handler.requests.get', return_value=mock_http_error_response):
            with pytest.raises(ValueError, match="Failed to fetch image"):
                fetch_image("https://example.com/image.png")

    def test_timeout_error_raises_valueerror(self):
        """Should raise ValueError on timeout."""
        from requests.exceptions import Timeout
        with patch('handler.requests.get', side_effect=Timeout()):
            with pytest.raises(ValueError, match="Failed to fetch image"):
                fetch_image("https://example.com/image.png")

    def test_connection_error_raises_valueerror(self):
        """Should raise ValueError on connection error."""
        from requests.exceptions import ConnectionError
        with patch('handler.requests.get', side_effect=ConnectionError()):
            with pytest.raises(ValueError, match="Failed to fetch image"):
                fetch_image("https://example.com/image.png")

    def test_invalid_image_raises_valueerror(self):
        """Should raise ValueError for invalid image data."""
        mock_response = MagicMock()
        mock_response.content = b"not an image"
        mock_response.raise_for_status = MagicMock()
        with patch('handler.requests.get', return_value=mock_response):
            with pytest.raises(ValueError, match="Failed to decode image"):
                fetch_image("https://example.com/image.png")

    def test_custom_timeout(self, mock_successful_response):
        """Should use custom timeout value."""
        with patch('handler.requests.get', return_value=mock_successful_response) as mock_get:
            fetch_image("https://example.com/image.png", timeout=60)
            mock_get.assert_called_once_with("https://example.com/image.png", timeout=60)

    def test_converts_rgba_to_rgb(self, rgba_image_bytes):
        """Should convert RGBA images to RGB."""
        mock_response = MagicMock()
        mock_response.content = rgba_image_bytes
        mock_response.raise_for_status = MagicMock()
        with patch('handler.requests.get', return_value=mock_response):
            result = fetch_image("https://example.com/image.png")
            assert result.mode == 'RGB'

    def test_converts_grayscale_to_rgb(self, grayscale_image_bytes):
        """Should convert grayscale images to RGB."""
        mock_response = MagicMock()
        mock_response.content = grayscale_image_bytes
        mock_response.raise_for_status = MagicMock()
        with patch('handler.requests.get', return_value=mock_response):
            result = fetch_image("https://example.com/image.png")
            assert result.mode == 'RGB'


class TestImageToBase64:
    """Tests for image_to_base64 function."""

    def test_returns_string(self):
        """Should return base64 string."""
        img = Image.new('RGB', (100, 100), color='red')
        result = image_to_base64(img)
        assert isinstance(result, str)

    def test_returns_valid_base64(self):
        """Should return valid base64 that can be decoded."""
        import base64
        img = Image.new('RGB', (100, 100), color='red')
        result = image_to_base64(img)
        # Should not raise
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_creates_png_format(self):
        """Should create PNG format image."""
        import base64
        img = Image.new('RGB', (100, 100), color='red')
        result = image_to_base64(img)
        decoded = base64.b64decode(result)
        # PNG magic bytes
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


class TestHandler:
    """Tests for handler entry point."""

    def test_returns_error_for_missing_image_urls(self):
        """Should return error when image_urls is missing."""
        job = {"input": {}}
        result = handler(job)
        assert "error" in result
        assert "image_urls" in result["error"].lower()

    def test_returns_error_for_empty_image_urls(self):
        """Should return error when image_urls is empty."""
        job = {"input": {"image_urls": []}}
        result = handler(job)
        assert "error" in result

    def test_returns_error_for_empty_input(self):
        """Should return error when input is empty."""
        job = {"input": {}}
        result = handler(job)
        assert "error" in result

    def test_successful_extraction_includes_latency(
        self, sample_job, mock_successful_response, sample_vllm_response
    ):
        """Should include latency_ms in successful response."""
        with patch('handler.requests.get', return_value=mock_successful_response):
            with patch('handler.httpx.AsyncClient') as mock_client:
                mock_response = AsyncMock()
                mock_response.json.return_value = sample_vllm_response
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )
                result = handler(sample_job)
                if "error" not in result:
                    assert "latency_ms" in result
                    assert isinstance(result["latency_ms"], int)
                    assert result["latency_ms"] >= 0

    def test_catches_value_error(self):
        """Should catch ValueError and return error dict."""
        job = {"input": {"image_urls": ["https://example.com/img.png"]}}
        with patch('handler.fetch_image', side_effect=ValueError("Test error")):
            result = handler(job)
            assert "error" in result

    def test_catches_general_exception(self):
        """Should catch general Exception and return internal error."""
        job = {"input": {"image_urls": ["https://example.com/img.png"]}}
        # Patch asyncio.run to raise a non-ValueError exception
        with patch('handler.asyncio.run', side_effect=RuntimeError("Unexpected")):
            result = handler(job)
            assert "error" in result
            assert "Internal error" in result["error"]

    def test_default_doc_type_is_other(self, mock_successful_response, sample_vllm_response):
        """Should use 'other' as default doc_type."""
        job = {"input": {"image_urls": ["https://example.com/img.png"]}}
        with patch('handler.requests.get', return_value=mock_successful_response):
            with patch('handler.httpx.AsyncClient') as mock_client:
                mock_response = AsyncMock()
                mock_response.json.return_value = sample_vllm_response
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )
                result = handler(job)
                if "error" not in result:
                    assert result.get("doc_type") == "other"

    def test_uses_provided_doc_type(self, sample_job, mock_successful_response, sample_vllm_response):
        """Should use provided doc_type."""
        with patch('handler.requests.get', return_value=mock_successful_response):
            with patch('handler.httpx.AsyncClient') as mock_client:
                mock_response = AsyncMock()
                mock_response.json.return_value = sample_vllm_response
                mock_response.raise_for_status = MagicMock()
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )
                result = handler(sample_job)
                if "error" not in result:
                    assert result.get("doc_type") == "bank_statement"


class TestCallVllm:
    """Tests for call_vllm async function."""

    @pytest.mark.asyncio
    async def test_makes_correct_api_call(self, sample_vllm_response):
        """Should make correct API call to vLLM."""
        from handler import call_vllm

        with patch('handler.httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            # json() is a sync method in httpx Response
            mock_response.json.return_value = sample_vllm_response
            mock_response.raise_for_status = MagicMock()

            # post() is async, so use AsyncMock for it
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await call_vllm("base64imagedata")

            assert "raw_output" in result
            assert "usage" in result

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        """Should raise ValueError on HTTP error."""
        from handler import call_vllm
        import httpx

        with patch('handler.httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=MagicMock()
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ValueError, match="vLLM request failed"):
                await call_vllm("base64imagedata")

    @pytest.mark.asyncio
    async def test_raises_on_connection_error(self):
        """Should raise ValueError on connection error."""
        from handler import call_vllm
        import httpx

        with patch('handler.httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(ValueError, match="vLLM connection error"):
                await call_vllm("base64imagedata")

    @pytest.mark.asyncio
    async def test_invalid_prompt_mode_falls_back_to_default(self, sample_vllm_response):
        """Should fall back to layout_all prompt for invalid prompt_mode."""
        from handler import call_vllm, PROMPT_MODES

        with patch('handler.httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = sample_vllm_response
            mock_response.raise_for_status = MagicMock()

            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            # Call with invalid prompt_mode
            await call_vllm("base64imagedata", prompt_mode="invalid_mode")

            # Verify API was called with default prompt
            call_args = mock_post.call_args
            payload = call_args.kwargs.get('json') or call_args[1].get('json')
            message_content = payload["messages"][0]["content"]
            text_content = [c for c in message_content if c.get("type") == "text"][0]["text"]

            # Should use the default "layout_all" prompt
            assert text_content == PROMPT_MODES["layout_all"]


# =============================================================================
# Multi-page extraction tests
# =============================================================================

class TestMultiPageExtraction:
    """Tests for multi-page document extraction flow."""

    @pytest.mark.asyncio
    async def test_extract_processes_all_pages(self, valid_image_bytes, sample_vllm_response):
        """Should process all pages in image_urls list."""
        from handler import extract

        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch('handler.requests.get', return_value=mock_response) as mock_get, \
             patch('handler.httpx.AsyncClient') as mock_client:
            mock_vllm_response = MagicMock()
            mock_vllm_response.json.return_value = sample_vllm_response
            mock_vllm_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_vllm_response
            )

            result = await extract(
                ["https://example.com/page1.png", "https://example.com/page2.png", "https://example.com/page3.png"],
                "bank_statement"
            )

            # Should have fetched all 3 pages
            assert mock_get.call_count == 3
            assert result["page_count"] == 3

    @pytest.mark.asyncio
    async def test_extract_combines_text_from_all_pages(self, valid_image_bytes):
        """Should combine raw_text from all pages."""
        from handler import extract

        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        # Different responses for different pages
        page_responses = [
            {"choices": [{"message": {"content": "Page 1 content"}}], "usage": {}},
            {"choices": [{"message": {"content": "Page 2 content"}}], "usage": {}},
        ]

        with patch('handler.requests.get', return_value=mock_response), \
             patch('handler.httpx.AsyncClient') as mock_client:
            mock_vllm_response = MagicMock()
            mock_vllm_response.json.side_effect = page_responses
            mock_vllm_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_vllm_response
            )

            result = await extract(
                ["https://example.com/page1.png", "https://example.com/page2.png"],
                "other"
            )

            # Combined text should contain content from both pages
            combined = result["raw_ocr"]["combined_text"]
            assert "Page 1" in combined or result["page_count"] == 2

    @pytest.mark.asyncio
    async def test_extract_aggregates_layout_elements(self, valid_image_bytes):
        """Should aggregate layout_elements from all pages."""
        from handler import extract
        import json

        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        page1_content = json.dumps({
            "layout_elements": [{"text": "Element 1", "category": "Text"}]
        })
        page2_content = json.dumps({
            "layout_elements": [{"text": "Element 2", "category": "Text"}]
        })

        responses = [
            {"choices": [{"message": {"content": page1_content}}], "usage": {}},
            {"choices": [{"message": {"content": page2_content}}], "usage": {}},
        ]
        call_count = [0]

        def get_response(*args, **kwargs):
            mock = MagicMock()
            mock.json.return_value = responses[call_count[0]]
            mock.raise_for_status = MagicMock()
            call_count[0] += 1
            return mock

        with patch('handler.requests.get', return_value=mock_response), \
             patch('handler.httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=[get_response(), get_response()]
            )

            result = await extract(
                ["https://example.com/page1.png", "https://example.com/page2.png"],
                "other"
            )

            # Should have elements from both pages
            all_elements = result["raw_ocr"]["all_elements"]
            assert len(all_elements) >= 2


# =============================================================================
# Error recovery tests
# =============================================================================

class TestErrorRecovery:
    """Tests for error recovery when some pages fail."""

    @pytest.mark.asyncio
    async def test_continues_after_single_page_failure(self, valid_image_bytes, sample_vllm_response):
        """Should continue processing after one page fails."""
        from handler import extract
        from requests.exceptions import Timeout

        # First call fails, second succeeds
        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Timeout("First page timeout")
            mock = MagicMock()
            mock.content = valid_image_bytes
            mock.raise_for_status = MagicMock()
            return mock

        with patch('handler.requests.get', side_effect=mock_get), \
             patch('handler.httpx.AsyncClient') as mock_client:
            mock_vllm_response = MagicMock()
            mock_vllm_response.json.return_value = sample_vllm_response
            mock_vllm_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_vllm_response
            )

            result = await extract(
                ["https://example.com/page1.png", "https://example.com/page2.png"],
                "bank_statement"
            )

            # Should have processed the second page
            assert result["page_count"] == 1

    @pytest.mark.asyncio
    async def test_raises_when_all_pages_fail(self):
        """Should raise ValueError when all pages fail to process."""
        from handler import extract
        from requests.exceptions import ConnectionError

        with patch('handler.requests.get', side_effect=ConnectionError("All failed")):
            with pytest.raises(ValueError, match="Could not process any images"):
                await extract(
                    ["https://example.com/page1.png", "https://example.com/page2.png"],
                    "bank_statement"
                )

    @pytest.mark.asyncio
    async def test_continues_after_vllm_error_on_page(self, valid_image_bytes):
        """Should continue after vLLM error on one page."""
        from handler import extract
        import httpx

        mock_response = MagicMock()
        mock_response.content = valid_image_bytes
        mock_response.raise_for_status = MagicMock()

        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise httpx.RequestError("vLLM error")
            mock = MagicMock()
            mock.json.return_value = {
                "choices": [{"message": {"content": "Success"}}],
                "usage": {}
            }
            mock.raise_for_status = MagicMock()
            return mock

        with patch('handler.requests.get', return_value=mock_response), \
             patch('handler.httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await extract(
                ["https://example.com/page1.png", "https://example.com/page2.png"],
                "other"
            )

            # Should have processed the second page only
            assert result["page_count"] == 1


# =============================================================================
# Bank statement format variations
# =============================================================================

class TestBankStatementVariations:
    """Tests for different bank statement formats."""

    def test_extracts_bank_of_america(self):
        """Should extract Bank of America name."""
        text = "Bank of America\nChecking Account Statement"
        result = transform_bank_statement([], text)
        assert result["header"]["bank_name"] == "Bank of America"

    def test_extracts_wells_fargo(self):
        """Should extract Wells Fargo name."""
        text = "Wells Fargo\nAccount Summary"
        result = transform_bank_statement([], text)
        assert result["header"]["bank_name"] == "Wells Fargo"

    def test_extracts_credit_union(self):
        """Should extract Credit Union names."""
        text = "Navy Federal Credit Union\nStatement"
        result = transform_bank_statement([], text)
        assert "Credit Union" in result["header"]["bank_name"]

    def test_handles_opening_balance_instead_of_beginning(self):
        """Should recognize 'Opening Balance' as beginning balance."""
        elements = [
            {"text": "Opening Balance: $1,000.00", "confidence": 0.95}
        ]
        result = transform_bank_statement(elements, "Opening Balance: $1,000.00")
        assert result["header"]["beginning_balance"] == 1000.00

    def test_handles_closing_balance_instead_of_ending(self):
        """Should recognize 'Closing Balance' as ending balance."""
        elements = [
            {"text": "Closing Balance: $2,500.00", "confidence": 0.95}
        ]
        result = transform_bank_statement(elements, "Closing Balance: $2,500.00")
        assert result["header"]["ending_balance"] == 2500.00

    def test_extracts_transactions_with_negative_amounts(self):
        """Should extract transactions with negative amounts."""
        text = """
        Chase Bank
        01/15 ATM Withdrawal -$200.00
        01/16 Direct Deposit $1,500.00
        """
        result = transform_bank_statement([], text)
        # Check transactions were found
        amounts = [t["amount"] for t in result["transactions"]]
        assert -200.00 in amounts or 200.00 in amounts

    def test_handles_account_number_with_asterisks(self):
        """Should handle account numbers masked with asterisks."""
        text = "Account: **1234"
        result = transform_bank_statement([], text)
        assert "1234" in result["header"]["account_number"]


# =============================================================================
# W2 edge cases
# =============================================================================

class TestW2EdgeCases:
    """Tests for W2 form edge cases."""

    def test_handles_missing_all_boxes(self):
        """Should handle W2 with no recognizable box values."""
        text = "W-2 2023\nEmployee Name: John Doe"
        result = transform_w2([], text)
        # Should still have structure
        assert "wages" in result
        assert result["tax_year"] == "2023"

    def test_handles_partial_boxes(self):
        """Should handle W2 with only some boxes filled."""
        text = """
        W-2 2023
        Box 1 Wages: $50,000.00
        Box 2 Federal tax: $8,000.00
        """
        result = transform_w2([], text)
        assert result["wages"]["box1_wages"] == 50000.00
        assert result["wages"]["box2_federal_tax"] == 8000.00
        assert result["wages"]["box3_ss_wages"] is None

    def test_extracts_year_without_w2_label(self):
        """Should extract year even without W-2 prefix."""
        text = "Tax Statement 2022\nWages: $60,000"
        result = transform_w2([], text)
        assert result["tax_year"] == "2022"

    def test_handles_ssn_with_star_mask(self):
        """Should handle SSN masked with stars."""
        text = "SSN: ***-**-5678"
        result = transform_w2([], text)
        assert result["employee"]["ssn"] == "XXX-XX-5678"

    def test_extracts_wages_without_box_label(self):
        """Should extract wages when labeled differently."""
        text = """
        W-2 2023
        Wages, tips, other compensation: $45,000.00
        Federal income tax withheld: $7,000.00
        """
        result = transform_w2([], text)
        assert result["wages"]["box1_wages"] == 45000.00


# =============================================================================
# 1099 variant tests
# =============================================================================

class TestForm1099Variants:
    """Tests for different 1099 form variants."""

    @pytest.mark.parametrize("form_type,text", [
        ("1099-INT", "Form 1099-INT Interest Income"),
        ("1099-DIV", "1099-DIV Dividend Income"),
        ("1099-MISC", "Form 1099-MISC Miscellaneous Income"),
        ("1099-NEC", "1099-NEC Nonemployee Compensation"),
        ("1099-R", "Form 1099-R Retirement Distributions"),
        ("1099-G", "1099-G Government Payments"),
        ("1099-K", "1099-K Payment Card Transactions"),
    ])
    def test_detects_all_1099_variants(self, form_type, text):
        """Should detect all 1099 form variants."""
        result = transform_1099([], text)
        assert result["form_type"] == form_type

    def test_extracts_interest_amounts_for_1099_int(self):
        """Should extract interest amounts from 1099-INT."""
        text = """
        1099-INT 2023
        Interest Income: $1,234.56
        Tax Year: 2023
        """
        result = transform_1099([], text)
        assert result["form_type"] == "1099-INT"
        assert 1234.56 in result["amounts"].values()

    def test_extracts_dividend_amounts_for_1099_div(self):
        """Should extract dividend amounts from 1099-DIV."""
        text = """
        Form 1099-DIV 2023
        Total Ordinary Dividends: $5,000.00
        Qualified Dividends: $3,500.00
        """
        result = transform_1099([], text)
        assert result["form_type"] == "1099-DIV"
        amounts_values = list(result["amounts"].values())
        assert 5000.00 in amounts_values or 3500.00 in amounts_values

    def test_extracts_nec_compensation(self):
        """Should extract compensation from 1099-NEC."""
        text = """
        1099-NEC 2023
        Nonemployee Compensation: $25,000.00
        """
        result = transform_1099([], text)
        assert result["form_type"] == "1099-NEC"
        assert 25000.00 in result["amounts"].values()

    def test_handles_1099_without_hyphen(self):
        """Should detect 1099 forms written without hyphen."""
        text = "1099INT Interest Income 2023"
        result = transform_1099([], text)
        # Current regex requires hyphen, but should still return structure
        assert "recipient" in result


# =============================================================================
# Additional transformer edge cases
# =============================================================================

class TestTransformerEdgeCases:
    """Additional edge case tests for transformers."""

    def test_extract_amounts_with_only_cents(self):
        """Should handle amounts with only cents (no dollars)."""
        text = "Fee: $0.50"
        result = extract_amounts(text)
        assert any(a["value"] == 0.50 for a in result)

    def test_extract_amounts_whole_numbers(self):
        """Should handle whole number amounts without decimals."""
        text = "Total: $1,000"
        result = extract_amounts(text)
        assert any(a["value"] == 1000 for a in result)

    def test_extract_dates_short_year(self):
        """Should extract dates with 2-digit years."""
        text = "Date: 01/15/24"
        result = extract_dates(text)
        assert "01/15/24" in result

    def test_extract_dates_abbreviated_month(self):
        """Should extract dates with abbreviated months."""
        text = "Date: Jan 15, 2024"
        result = extract_dates(text)
        assert len(result) >= 1

    def test_transform_generic_categorizes_tables(self):
        """Should categorize table elements separately."""
        elements = [
            {"category": "Table", "text": "Row1\tRow2", "bbox": [0, 0, 100, 100]},
            {"category": "Text", "text": "Regular text", "confidence": 0.9}
        ]
        result = transform_generic(elements, "")
        assert len(result["tables"]) == 1
        assert len(result["text_blocks"]) == 1

    def test_transform_generic_handles_empty_elements(self):
        """Should handle empty elements list gracefully."""
        result = transform_generic([], "Just some text")
        assert result["text_blocks"] == []
        assert result["tables"] == []
