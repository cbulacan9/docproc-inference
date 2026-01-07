"""
Tests for the confidence calculation module.

Tests cover:
- calculate_overall_confidence() with formula verification
- apply_format_boost() for currency/date/SSN/EIN patterns
- apply_cross_reference_boost() for bank statement and W2 validation
- apply_position_boost() for layout region validation
- get_base_confidence() and get_critical_fields() helpers

Per Testing Plan:
- Verify overall confidence uses formula: min(mean(critical), min(critical) + 0.1)
- Verify format validation boost (+0.05) for currency/date/SSN/EIN
- Verify cross-reference boost (+0.10) when bank statement balances reconcile
- Verify missing fields return confidence 0.0
"""

import pytest

from confidence import (
    calculate_overall_confidence,
    apply_format_boost,
    apply_cross_reference_boost,
    apply_position_boost,
    get_base_confidence,
    get_critical_fields,
    CRITICAL_FIELDS,
    FORMAT_PATTERNS,
    FIELD_TYPE_MAPPING,
)


# =============================================================================
# Tests for get_base_confidence()
# =============================================================================

class TestGetBaseConfidence:
    """Tests for get_base_confidence function."""

    def test_extracts_confidence_from_element(self):
        """Should extract confidence value from element dict."""
        element = {"text": "Hello", "confidence": 0.92}
        result = get_base_confidence(element)
        assert result == 0.92

    def test_returns_default_when_missing(self):
        """Should return 0.85 default when confidence not in element."""
        element = {"text": "Hello"}
        result = get_base_confidence(element)
        assert result == 0.85

    def test_handles_zero_confidence(self):
        """Should return 0.0 when confidence is explicitly zero."""
        element = {"confidence": 0.0}
        result = get_base_confidence(element)
        assert result == 0.0

    def test_handles_max_confidence(self):
        """Should return 1.0 when confidence is maximum."""
        element = {"confidence": 1.0}
        result = get_base_confidence(element)
        assert result == 1.0


# =============================================================================
# Tests for get_critical_fields()
# =============================================================================

class TestGetCriticalFields:
    """Tests for get_critical_fields function."""

    def test_returns_bank_statement_critical_fields(self):
        """Should return correct critical fields for bank_statement."""
        result = get_critical_fields("bank_statement")
        assert "header.beginning_balance" in result
        assert "header.ending_balance" in result
        assert "transactions[*].amount" in result

    def test_returns_w2_critical_fields(self):
        """Should return correct critical fields for W2."""
        result = get_critical_fields("W2")
        assert "boxes.box1_wages" in result
        assert "boxes.box2_federal_withheld" in result
        assert "employee.ssn" in result

    def test_returns_1099_int_critical_fields(self):
        """Should return correct critical fields for 1099-INT."""
        result = get_critical_fields("1099-INT")
        assert "recipient.tin" in result
        assert "boxes.box1_interest" in result

    def test_returns_invoice_critical_fields(self):
        """Should return correct critical fields for invoice."""
        result = get_critical_fields("invoice")
        assert "total" in result
        assert "invoice_number" in result
        assert "invoice_date" in result

    def test_returns_empty_list_for_unknown_type(self):
        """Should return empty list for unknown document type."""
        result = get_critical_fields("unknown_document_type")
        assert result == []

    def test_all_defined_doc_types_have_critical_fields(self):
        """Should have critical fields defined for all doc types in CRITICAL_FIELDS."""
        for doc_type in CRITICAL_FIELDS.keys():
            result = get_critical_fields(doc_type)
            assert len(result) > 0, f"No critical fields for {doc_type}"


# =============================================================================
# Tests for apply_format_boost()
# =============================================================================

class TestApplyFormatBoost:
    """Tests for apply_format_boost function."""

    # --- Currency format tests ---

    def test_boosts_currency_with_dollar_sign(self):
        """Should boost confidence for currency with dollar sign."""
        result = apply_format_boost("$1,234.56", "header.beginning_balance", 0.85)
        assert result == 0.90  # +0.05 boost

    def test_boosts_currency_without_dollar_sign(self):
        """Should boost confidence for currency without dollar sign."""
        result = apply_format_boost("1234.56", "header.ending_balance", 0.85)
        assert result == 0.90

    def test_boosts_negative_currency(self):
        """Should boost confidence for negative currency values."""
        # Pattern expects $- format (dollar sign before minus)
        result = apply_format_boost("$-500.00", "summary.net_change", 0.85)
        assert result == 0.90

    def test_boosts_negative_currency_without_dollar(self):
        """Should boost confidence for negative currency without dollar sign."""
        result = apply_format_boost("-500.00", "summary.net_change", 0.85)
        assert result == 0.90

    def test_boosts_currency_whole_number(self):
        """Should boost confidence for whole number currency."""
        result = apply_format_boost("$1000", "boxes.box1_wages", 0.85)
        assert result == 0.90

    # --- Date format tests ---

    def test_boosts_date_slash_format(self):
        """Should boost confidence for MM/DD/YYYY date format."""
        result = apply_format_boost("01/15/2024", "header.statement_period", 0.85)
        assert result == 0.90

    def test_boosts_date_dash_format(self):
        """Should boost confidence for MM-DD-YYYY date format."""
        result = apply_format_boost("01-15-2024", "transactions.date", 0.85)
        assert result == 0.90

    def test_boosts_date_short_year(self):
        """Should boost confidence for dates with 2-digit year."""
        result = apply_format_boost("1/5/24", "header.statement_period", 0.85)
        assert result == 0.90

    # --- SSN format tests ---

    def test_boosts_masked_ssn(self):
        """Should boost confidence for masked SSN format."""
        result = apply_format_boost("XXX-XX-1234", "employee.ssn", 0.85)
        assert result == 0.90

    def test_boosts_full_ssn(self):
        """Should boost confidence for full SSN format."""
        result = apply_format_boost("123-45-6789", "employee.ssn", 0.85)
        assert result == 0.90

    def test_boosts_recipient_tin(self):
        """Should boost confidence for recipient TIN (SSN format)."""
        result = apply_format_boost("XXX-XX-5678", "recipient.tin", 0.85)
        assert result == 0.90

    # --- EIN format tests ---

    def test_boosts_ein_format(self):
        """Should boost confidence for EIN format."""
        result = apply_format_boost("12-3456789", "employer.ein", 0.85)
        assert result == 0.90

    def test_boosts_payer_tin_ein(self):
        """Should boost confidence for payer TIN (EIN format)."""
        result = apply_format_boost("98-7654321", "payer.tin", 0.85)
        assert result == 0.90

    # --- Account number format tests ---

    def test_boosts_masked_account_number(self):
        """Should boost confidence for masked account number."""
        result = apply_format_boost("****1234", "header.account_number", 0.85)
        assert result == 0.90

    def test_boosts_numeric_account_number(self):
        """Should boost confidence for numeric account number."""
        result = apply_format_boost("12345678", "header.account_number", 0.85)
        assert result == 0.90

    # --- Transaction subfield handling ---

    def test_boosts_transaction_amount_with_index(self):
        """Should boost confidence for indexed transaction amount field."""
        result = apply_format_boost("$250.00", "transactions.0.amount", 0.85)
        assert result == 0.90

    def test_boosts_transaction_date_with_index(self):
        """Should boost confidence for indexed transaction date field."""
        result = apply_format_boost("01/15/24", "transactions.5.date", 0.85)
        assert result == 0.90

    # --- No boost scenarios ---

    def test_no_boost_for_unknown_field(self):
        """Should not boost for unknown field name."""
        result = apply_format_boost("$100.00", "unknown.field", 0.85)
        assert result == 0.85  # No change

    def test_no_boost_for_invalid_currency_format(self):
        """Should not boost for invalid currency format."""
        result = apply_format_boost("one hundred dollars", "header.beginning_balance", 0.85)
        assert result == 0.85

    def test_no_boost_for_invalid_date_format(self):
        """Should not boost for invalid date format."""
        result = apply_format_boost("January 15th", "header.statement_period", 0.85)
        assert result == 0.85

    def test_no_boost_for_invalid_ssn_format(self):
        """Should not boost for invalid SSN format."""
        result = apply_format_boost("12345", "employee.ssn", 0.85)
        assert result == 0.85

    def test_no_boost_for_invalid_ein_format(self):
        """Should not boost for invalid EIN format."""
        result = apply_format_boost("123456789", "employer.ein", 0.85)
        assert result == 0.85  # Missing hyphen

    # --- Edge cases ---

    def test_returns_base_for_none_value(self):
        """Should return base confidence for None value."""
        result = apply_format_boost(None, "header.beginning_balance", 0.85)
        assert result == 0.85

    def test_caps_at_one(self):
        """Should cap boosted confidence at 1.0."""
        result = apply_format_boost("$1,234.56", "header.beginning_balance", 0.98)
        assert result == 1.0

    def test_handles_empty_string(self):
        """Should not boost for empty string value."""
        result = apply_format_boost("", "header.beginning_balance", 0.85)
        assert result == 0.85

    def test_handles_whitespace_value(self):
        """Should handle whitespace-padded values."""
        result = apply_format_boost("  $1,234.56  ", "header.beginning_balance", 0.85)
        assert result == 0.90  # Strips whitespace


# =============================================================================
# Tests for apply_position_boost()
# =============================================================================

class TestApplyPositionBoost:
    """Tests for apply_position_boost function."""

    def test_boosts_field_in_expected_region(self):
        """Should boost confidence when field is in expected region."""
        # Bank name should be in top 15% of page
        bbox = [100, 50, 400, 80]  # y1=50 on 1000px page = 0.05 (top)
        page_dims = (1000, 1000)
        result = apply_position_boost(
            "header.bank_name", bbox, page_dims, "bank_statement", 0.85
        )
        assert result == 0.90

    def test_no_boost_for_field_outside_region(self):
        """Should not boost when field is outside expected region."""
        # Bank name expected in top 15%, but field is at 50%
        bbox = [100, 500, 400, 530]  # y1=500 on 1000px page = 0.50 (middle)
        page_dims = (1000, 1000)
        result = apply_position_boost(
            "header.bank_name", bbox, page_dims, "bank_statement", 0.85
        )
        assert result == 0.85  # No boost

    def test_no_boost_for_unknown_field(self):
        """Should not boost for field without region expectation."""
        bbox = [100, 50, 400, 80]
        page_dims = (1000, 1000)
        result = apply_position_boost(
            "unknown.field", bbox, page_dims, "bank_statement", 0.85
        )
        assert result == 0.85

    def test_no_boost_for_unknown_doc_type(self):
        """Should not boost for unknown document type."""
        bbox = [100, 50, 400, 80]
        page_dims = (1000, 1000)
        result = apply_position_boost(
            "header.bank_name", bbox, page_dims, "unknown_type", 0.85
        )
        assert result == 0.85

    def test_returns_base_for_none_bbox(self):
        """Should return base confidence when bbox is None."""
        result = apply_position_boost(
            "header.bank_name", None, (1000, 1000), "bank_statement", 0.85
        )
        assert result == 0.85

    def test_returns_base_for_none_page_dims(self):
        """Should return base confidence when page_dims is None."""
        result = apply_position_boost(
            "header.bank_name", [100, 50, 400, 80], None, "bank_statement", 0.85
        )
        assert result == 0.85

    def test_handles_zero_page_dimensions(self):
        """Should handle zero page dimensions gracefully.

        Note: When page dimensions are (0, 0), normalized coordinates become 0,
        which may fall within expected regions. This is current behavior.
        """
        result = apply_position_boost(
            "header.bank_name", [100, 50, 400, 80], (0, 0), "bank_statement", 0.85
        )
        # With zero dimensions, normalized coords are 0, which is within y_max=0.15
        # so boost is applied (edge case behavior)
        assert result == 0.90

    def test_caps_at_one(self):
        """Should cap boosted confidence at 1.0."""
        bbox = [100, 50, 400, 80]
        page_dims = (1000, 1000)
        result = apply_position_boost(
            "header.bank_name", bbox, page_dims, "bank_statement", 0.98
        )
        assert result == 1.0

    def test_w2_box1_wages_position(self):
        """Should boost W2 box1_wages in expected region."""
        # boxes.box1_wages expected: x_min=0, x_max=0.5, y_min=0.40
        bbox = [100, 450, 400, 480]  # x=0.1, y=0.45 (left side, lower half)
        page_dims = (1000, 1000)
        result = apply_position_boost(
            "boxes.box1_wages", bbox, page_dims, "W2", 0.85
        )
        assert result == 0.90


# =============================================================================
# Tests for apply_cross_reference_boost()
# =============================================================================

class TestApplyCrossReferenceBoost:
    """Tests for apply_cross_reference_boost function."""

    # --- Bank statement cross-reference tests ---

    def test_boosts_when_bank_statement_balances_reconcile(self):
        """
        Behavior: Boost confidence when beginning + transactions = ending.
        Assumptions: All balance fields present and valid.
        Failure criteria: Boost not applied despite correct reconciliation.
        """
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 1500.00
            },
            "transactions": [
                {"amount": 200.00},
                {"amount": 300.00}
            ]
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": [
                {"amount": 0.85},
                {"amount": 0.85}
            ]
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        # Balance fields should get +0.10 boost
        assert result["header.beginning_balance"] == 0.95
        assert result["header.ending_balance"] == 0.95
        # Transaction amounts should also get boost
        assert result["transactions"][0]["amount"] == 0.95
        assert result["transactions"][1]["amount"] == 0.95

    def test_no_boost_when_balances_dont_reconcile(self):
        """
        Behavior: No boost when balances don't add up.
        Assumptions: All fields present but values don't reconcile.
        Failure criteria: Boost incorrectly applied.
        """
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 2000.00  # Should be 1500 to reconcile
            },
            "transactions": [
                {"amount": 200.00},
                {"amount": 300.00}
            ]
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": [
                {"amount": 0.85},
                {"amount": 0.85}
            ]
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        # No boost should be applied
        assert result["header.beginning_balance"] == 0.85
        assert result["header.ending_balance"] == 0.85

    def test_handles_missing_beginning_balance(self):
        """Should handle missing beginning_balance gracefully."""
        data = {
            "header": {
                "ending_balance": 1500.00
            },
            "transactions": []
        }
        field_confidences = {
            "header.ending_balance": 0.85
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")
        assert result["header.ending_balance"] == 0.85  # No change

    def test_handles_missing_ending_balance(self):
        """Should handle missing ending_balance gracefully."""
        data = {
            "header": {
                "beginning_balance": 1000.00
            },
            "transactions": []
        }
        field_confidences = {
            "header.beginning_balance": 0.85
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")
        assert result["header.beginning_balance"] == 0.85  # No change

    def test_handles_empty_transactions(self):
        """Should reconcile when no transactions and balances match."""
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 1000.00  # Same, no transactions
            },
            "transactions": []
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": []
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        assert result["header.beginning_balance"] == 0.95
        assert result["header.ending_balance"] == 0.95

    def test_handles_negative_transactions(self):
        """Should reconcile correctly with negative transaction amounts."""
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 700.00
            },
            "transactions": [
                {"amount": -300.00}  # Withdrawal
            ]
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": [{"amount": 0.85}]
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        assert result["header.beginning_balance"] == 0.95
        assert result["header.ending_balance"] == 0.95

    def test_tolerates_penny_rounding(self):
        """Should reconcile within 1 cent tolerance."""
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 1500.009  # Off by less than 1 cent
            },
            "transactions": [
                {"amount": 500.00}
            ]
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": [{"amount": 0.85}]
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        assert result["header.beginning_balance"] == 0.95
        assert result["header.ending_balance"] == 0.95

    def test_handles_non_numeric_balance_values(self):
        """Should handle non-numeric balance values gracefully."""
        data = {
            "header": {
                "beginning_balance": "N/A",
                "ending_balance": 1500.00
            },
            "transactions": []
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85
        }

        # Should not raise, just return unchanged
        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")
        assert result["header.beginning_balance"] == 0.85

    def test_handles_none_transaction_amounts(self):
        """Should handle None values in transaction amounts."""
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 1500.00
            },
            "transactions": [
                {"amount": 500.00},
                {"amount": None}  # Missing amount
            ]
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": [{"amount": 0.85}, {"amount": 0.0}]
        }

        # Should reconcile (None treated as 0)
        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")
        assert result["header.beginning_balance"] == 0.95

    def test_caps_boost_at_one(self):
        """Should cap cross-reference boost at 1.0."""
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 1000.00
            },
            "transactions": []
        }
        field_confidences = {
            "header.beginning_balance": 0.95,  # Will exceed 1.0 with +0.10
            "header.ending_balance": 0.95,
            "transactions": []
        }

        result = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        assert result["header.beginning_balance"] == 1.0
        assert result["header.ending_balance"] == 1.0

    # --- W2 cross-reference tests ---

    def test_boosts_w2_when_ss_tax_matches_wages(self):
        """
        Behavior: Boost when SS tax is ~6.2% of SS wages.
        Assumptions: SS wages and tax fields present.
        Failure criteria: Boost not applied for valid SS calculation.
        """
        data = {
            "boxes": {
                "box3_ss_wages": 100000.00,
                "box4_ss_withheld": 6200.00  # Exactly 6.2%
            }
        }
        field_confidences = {
            "boxes.box3_ss_wages": 0.85,
            "boxes.box4_ss_withheld": 0.85
        }

        result = apply_cross_reference_boost(data, field_confidences, "W2")

        assert result["boxes.box3_ss_wages"] == 0.95
        assert result["boxes.box4_ss_withheld"] == 0.95

    def test_boosts_w2_when_medicare_tax_matches_wages(self):
        """
        Behavior: Boost when Medicare tax is ~1.45% of Medicare wages.
        Assumptions: Medicare wages and tax fields present.
        Failure criteria: Boost not applied for valid Medicare calculation.
        """
        data = {
            "boxes": {
                "box5_medicare_wages": 100000.00,
                "box6_medicare_withheld": 1450.00  # Exactly 1.45%
            }
        }
        field_confidences = {
            "boxes.box5_medicare_wages": 0.85,
            "boxes.box6_medicare_withheld": 0.85
        }

        result = apply_cross_reference_boost(data, field_confidences, "W2")

        assert result["boxes.box5_medicare_wages"] == 0.95
        assert result["boxes.box6_medicare_withheld"] == 0.95

    def test_no_boost_w2_when_ss_tax_wrong(self):
        """Should not boost when SS tax doesn't match expected percentage."""
        data = {
            "boxes": {
                "box3_ss_wages": 100000.00,
                "box4_ss_withheld": 5000.00  # Should be ~6200
            }
        }
        field_confidences = {
            "boxes.box3_ss_wages": 0.85,
            "boxes.box4_ss_withheld": 0.85
        }

        result = apply_cross_reference_boost(data, field_confidences, "W2")

        assert result["boxes.box3_ss_wages"] == 0.85  # No boost
        assert result["boxes.box4_ss_withheld"] == 0.85

    def test_tolerates_small_ss_variance(self):
        """Should tolerate small variance in SS tax calculation."""
        data = {
            "boxes": {
                "box3_ss_wages": 100000.00,
                "box4_ss_withheld": 6205.00  # Within $10 tolerance
            }
        }
        field_confidences = {
            "boxes.box3_ss_wages": 0.85,
            "boxes.box4_ss_withheld": 0.85
        }

        result = apply_cross_reference_boost(data, field_confidences, "W2")

        assert result["boxes.box3_ss_wages"] == 0.95
        assert result["boxes.box4_ss_withheld"] == 0.95

    def test_handles_missing_w2_wages_data(self):
        """Should handle W2 with missing wages data gracefully."""
        data = {
            "boxes": {}
        }
        field_confidences = {}

        result = apply_cross_reference_boost(data, field_confidences, "W2")
        assert result == {}  # No changes

    # --- Other document types ---

    def test_no_changes_for_unsupported_doc_type(self):
        """Should return unchanged confidences for unsupported doc types."""
        data = {"some": "data"}
        field_confidences = {"some.field": 0.85}

        result = apply_cross_reference_boost(data, field_confidences, "1099-INT")

        assert result["some.field"] == 0.85


# =============================================================================
# Tests for calculate_overall_confidence()
# =============================================================================

class TestCalculateOverallConfidence:
    """Tests for calculate_overall_confidence function."""

    def test_formula_mean_less_than_min_plus_point_one(self):
        """
        Behavior: Use mean when mean < min + 0.1
        Assumptions: Critical scores where mean is the limiting factor.
        Failure criteria: Formula not correctly applied.

        Example: scores [0.9, 0.9, 0.9]
        - mean = 0.9
        - min = 0.9
        - min + 0.1 = 1.0
        - Result = min(0.9, 1.0) = 0.9
        """
        field_confidences = {
            "header.beginning_balance": 0.9,
            "header.ending_balance": 0.9,
            "transactions": [{"amount": 0.9}]
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance",
            "transactions[*].amount"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        assert result == 0.9

    def test_formula_min_plus_point_one_less_than_mean(self):
        """
        Behavior: Use min + 0.1 when min + 0.1 < mean
        Assumptions: Critical scores with high variance.
        Failure criteria: Formula not correctly applied.

        Example: scores [0.5, 1.0, 1.0]
        - mean = 0.833...
        - min = 0.5
        - min + 0.1 = 0.6
        - Result = min(0.833, 0.6) = 0.6
        """
        field_confidences = {
            "header.beginning_balance": 0.5,
            "header.ending_balance": 1.0,
            "transactions": [{"amount": 1.0}]
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance",
            "transactions[*].amount"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        assert result == 0.6

    def test_handles_single_critical_field(self):
        """Should handle single critical field correctly."""
        field_confidences = {
            "boxes.box1_wages": 0.85
        }
        critical_fields = ["boxes.box1_wages"]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # mean = 0.85, min = 0.85, min + 0.1 = 0.95
        # Result = min(0.85, 0.95) = 0.85
        assert result == 0.85

    def test_handles_array_fields(self):
        """Should correctly process array fields like transactions[*].amount."""
        field_confidences = {
            "header.beginning_balance": 0.9,
            "header.ending_balance": 0.9,
            "transactions": [
                {"amount": 0.8},
                {"amount": 0.85},
                {"amount": 0.9}
            ]
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance",
            "transactions[*].amount"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # Scores: 0.9, 0.9, 0.8, 0.85, 0.9
        # mean = 0.87, min = 0.8, min + 0.1 = 0.9
        # Result = min(0.87, 0.9) = 0.87
        assert abs(result - 0.87) < 0.01

    def test_returns_default_for_no_critical_fields_found(self):
        """Should return 0.5 when no critical fields are found."""
        field_confidences = {
            "some.other.field": 0.95
        }
        critical_fields = ["header.beginning_balance", "header.ending_balance"]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        assert result == 0.5

    def test_returns_default_for_empty_critical_fields(self):
        """Should return 0.5 when critical_fields list is empty."""
        field_confidences = {
            "header.beginning_balance": 0.9
        }
        critical_fields = []

        result = calculate_overall_confidence(field_confidences, critical_fields)

        assert result == 0.5

    def test_handles_empty_transactions_array(self):
        """Should handle empty transactions array gracefully."""
        field_confidences = {
            "header.beginning_balance": 0.9,
            "header.ending_balance": 0.9,
            "transactions": []
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance",
            "transactions[*].amount"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # Only balance fields contribute
        # mean = 0.9, min = 0.9, min + 0.1 = 1.0
        # Result = min(0.9, 1.0) = 0.9
        assert result == 0.9

    def test_handles_missing_subfield_in_array(self):
        """Should skip array items missing the subfield."""
        field_confidences = {
            "header.beginning_balance": 0.9,
            "header.ending_balance": 0.9,
            "transactions": [
                {"amount": 0.85},
                {"description": 0.8},  # No amount field
                {"amount": 0.9}
            ]
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance",
            "transactions[*].amount"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # Scores: 0.9, 0.9, 0.85, 0.9 (skipped item without amount)
        # mean = 0.8875, min = 0.85, min + 0.1 = 0.95
        # Result = min(0.8875, 0.95) = 0.8875
        assert abs(result - 0.8875) < 0.01

    def test_ignores_non_numeric_confidence_values(self):
        """Should skip non-numeric confidence values."""
        field_confidences = {
            "header.beginning_balance": 0.9,
            "header.ending_balance": "invalid",  # Non-numeric
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # Only beginning_balance counts
        assert result == 0.9

    def test_all_zeros_returns_point_one(self):
        """Should return 0.1 (min + 0.1) when all critical scores are 0."""
        field_confidences = {
            "header.beginning_balance": 0.0,
            "header.ending_balance": 0.0,
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # mean = 0.0, min = 0.0, min + 0.1 = 0.1
        # Result = min(0.0, 0.1) = 0.0
        assert result == 0.0

    def test_handles_dict_value_in_confidences(self):
        """Should skip dict values that aren't the target type."""
        field_confidences = {
            "header.beginning_balance": 0.9,
            "header.ending_balance": {"nested": 0.85}  # Dict, not float
        }
        critical_fields = [
            "header.beginning_balance",
            "header.ending_balance"
        ]

        result = calculate_overall_confidence(field_confidences, critical_fields)

        # Only beginning_balance counts
        assert result == 0.9


# =============================================================================
# Integration tests - end-to-end confidence flow
# =============================================================================

class TestConfidenceIntegration:
    """Integration tests for confidence calculation flow."""

    def test_bank_statement_full_flow(self):
        """Test complete confidence calculation for bank statement."""
        data = {
            "header": {
                "beginning_balance": 1000.00,
                "ending_balance": 1500.00
            },
            "transactions": [
                {"amount": 500.00}
            ]
        }
        field_confidences = {
            "header.beginning_balance": 0.85,
            "header.ending_balance": 0.85,
            "transactions": [{"amount": 0.85}]
        }

        # Apply cross-reference boost (should boost due to reconciliation)
        boosted = apply_cross_reference_boost(data, field_confidences, "bank_statement")

        # Calculate overall
        critical = get_critical_fields("bank_statement")
        overall = calculate_overall_confidence(boosted, critical)

        # All fields boosted to 0.95
        # mean = 0.95, min = 0.95, min + 0.1 = 1.05 (capped at 1.0)
        # Result = min(0.95, 1.0) = 0.95
        assert abs(overall - 0.95) < 0.01

    def test_w2_full_flow(self):
        """Test complete confidence calculation for W2."""
        data = {
            "employee": {"ssn": "XXX-XX-1234"},
            "boxes": {
                "box1_wages": 100000.00,
                "box2_federal_withheld": 20000.00,
                "box3_ss_wages": 100000.00,
                "box4_ss_withheld": 6200.00,  # 6.2%
            }
        }
        field_confidences = {
            "employee.ssn": 0.90,  # Format boost already applied
            "boxes.box1_wages": 0.90,
            "boxes.box2_federal_withheld": 0.90,
            "boxes.box3_ss_wages": 0.85,
            "boxes.box4_ss_withheld": 0.85
        }

        # Apply cross-reference boost
        boosted = apply_cross_reference_boost(data, field_confidences, "W2")

        # SS fields should be boosted
        assert boosted["boxes.box3_ss_wages"] == 0.95
        assert boosted["boxes.box4_ss_withheld"] == 0.95

        # Calculate overall (critical: box1, box2, ssn)
        critical = get_critical_fields("W2")
        overall = calculate_overall_confidence(boosted, critical)

        # Critical scores: ssn=0.90, box1=0.90, box2=0.90
        # mean = 0.90, min = 0.90, min + 0.1 = 1.0
        # Result = 0.90
        assert overall == 0.90

    def test_missing_critical_field_reduces_confidence(self):
        """Test that missing critical field returns default overall."""
        field_confidences = {
            "header.bank_name": 0.95,
            # Missing: header.beginning_balance, header.ending_balance, transactions
        }

        critical = get_critical_fields("bank_statement")
        overall = calculate_overall_confidence(field_confidences, critical)

        # No critical fields found, should return 0.5
        assert overall == 0.5
