"""
Confidence calculation for document extraction.

This module implements per-field confidence scoring as specified in
the Extraction Confidence Contract.

Confidence is derived from:
1. Base OCR element confidence from dots.ocr
2. Format validation boost (+0.05) when values match expected patterns
3. Cross-reference validation boost (+0.10) when fields cross-validate
4. Layout position boost (+0.05) when fields appear in expected regions
"""

import re
from typing import Any


# Critical fields by document type - used for overall confidence calculation
CRITICAL_FIELDS: dict[str, list[str]] = {
    "bank_statement": [
        "header.beginning_balance",
        "header.ending_balance",
        "transactions[*].amount",
    ],
    "W2": [
        "boxes.box1_wages",
        "boxes.box2_federal_withheld",
        "employee.ssn",
    ],
    "1099-INT": ["recipient.tin", "boxes.box1_interest"],
    "1099-DIV": ["recipient.tin", "boxes.box1a_dividends"],
    "1099-MISC": ["recipient.tin", "boxes.box7_nonemployee"],
    "1099-NEC": ["recipient.tin", "boxes.box1_nonemployee"],
    "1099-R": ["recipient.tin", "boxes.box1_gross_distribution"],
    "invoice": ["total", "invoice_number", "invoice_date"],
}

# Format patterns for validation boost
FORMAT_PATTERNS: dict[str, str] = {
    "date": r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
    "currency": r"^\$?-?[\d,]+\.?\d{0,2}$",
    "ssn": r"^(\d{3}-\d{2}-\d{4}|XXX-XX-\d{4})$",
    "ein": r"^\d{2}-\d{7}$",
    "account_number": r"^[*\d]{4,}$",
}

# Field type mappings for format validation
FIELD_TYPE_MAPPING: dict[str, str] = {
    # Bank statement fields
    "header.beginning_balance": "currency",
    "header.ending_balance": "currency",
    "header.account_number": "account_number",
    "header.statement_period": "date",
    "summary.total_credits": "currency",
    "summary.total_debits": "currency",
    "summary.net_change": "currency",
    "transactions.date": "date",
    "transactions.amount": "currency",
    # W2 fields
    "employee.ssn": "ssn",
    "employer.ein": "ein",
    "boxes.box1_wages": "currency",
    "boxes.box2_federal_withheld": "currency",
    "boxes.box3_ss_wages": "currency",
    "boxes.box4_ss_withheld": "currency",
    "boxes.box5_medicare_wages": "currency",
    "boxes.box6_medicare_withheld": "currency",
    # 1099 fields
    "recipient.tin": "ssn",
    "payer.tin": "ein",
}

# Expected regions for layout position boost (normalized 0-1 coordinates)
EXPECTED_REGIONS: dict[str, dict[str, dict[str, float]]] = {
    "bank_statement": {
        "header.bank_name": {"y_max": 0.15},
        "header.account_number": {"y_max": 0.20},
        "header.statement_period": {"y_max": 0.25},
        "header.beginning_balance": {"y_max": 0.30},
        "header.ending_balance": {"y_min": 0.70},
        "transactions": {"y_min": 0.25, "y_max": 0.85},
    },
    "W2": {
        "employer.name": {"y_max": 0.25},
        "employer.ein": {"y_max": 0.25},
        "employee.ssn": {"y_max": 0.35},
        "employee.name": {"y_max": 0.40},
        "boxes.box1_wages": {"x_min": 0.0, "x_max": 0.5, "y_min": 0.40},
    },
}


def get_base_confidence(element: dict) -> float:
    """
    Extract confidence from dots.ocr element.

    Args:
        element: Layout element dict from dots.ocr

    Returns:
        Confidence score (default 0.85 if not provided)
    """
    return element.get("confidence", 0.85)


def apply_format_boost(
    value: Any,
    field_name: str,
    base_conf: float
) -> float:
    """
    Boost confidence if value matches expected format.

    Args:
        value: Field value to validate
        field_name: Dot-notation field name
        base_conf: Base confidence score

    Returns:
        Boosted confidence (max +0.05)
    """
    if value is None:
        return base_conf

    # Get field type from mapping
    field_type = FIELD_TYPE_MAPPING.get(field_name)

    # Also check for transaction subfields
    if not field_type and "." in field_name:
        # e.g., "transactions.0.amount" -> check "transactions.amount"
        parts = field_name.split(".")
        generic_key = f"{parts[0]}.{parts[-1]}"
        field_type = FIELD_TYPE_MAPPING.get(generic_key)

    if not field_type:
        return base_conf

    pattern = FORMAT_PATTERNS.get(field_type)
    if not pattern:
        return base_conf

    str_value = str(value).strip()
    if re.match(pattern, str_value):
        return min(base_conf + 0.05, 1.0)

    return base_conf


def apply_position_boost(
    field_name: str,
    bbox: list[int] | None,
    page_dims: tuple[int, int] | None,
    doc_type: str,
    base_conf: float
) -> float:
    """
    Boost confidence if field found in expected region.

    Args:
        field_name: Dot-notation field name
        bbox: Bounding box [x1, y1, x2, y2]
        page_dims: Page dimensions (width, height)
        doc_type: Document type
        base_conf: Base confidence score

    Returns:
        Boosted confidence (max +0.05)
    """
    if not bbox or not page_dims:
        return base_conf

    regions = EXPECTED_REGIONS.get(doc_type, {})
    expected = regions.get(field_name)

    if not expected:
        return base_conf

    page_width, page_height = page_dims

    # Normalize bbox coordinates
    x_normalized = bbox[0] / page_width if page_width > 0 else 0
    y_normalized = bbox[1] / page_height if page_height > 0 else 0

    # Check if within expected region
    x_ok = (
        expected.get("x_min", 0) <= x_normalized <= expected.get("x_max", 1)
    )
    y_ok = (
        expected.get("y_min", 0) <= y_normalized <= expected.get("y_max", 1)
    )

    if x_ok and y_ok:
        return min(base_conf + 0.05, 1.0)

    return base_conf


def apply_cross_reference_boost(
    data: dict[str, Any],
    field_confidences: dict[str, Any],
    doc_type: str
) -> dict[str, Any]:
    """
    Boost confidence for fields that cross-validate.

    Args:
        data: Extracted document data
        field_confidences: Current field confidence scores
        doc_type: Document type

    Returns:
        Updated field confidences with cross-reference boosts
    """
    if doc_type == "bank_statement":
        field_confidences = _cross_ref_bank_statement(data, field_confidences)
    elif doc_type == "W2":
        field_confidences = _cross_ref_w2(data, field_confidences)

    return field_confidences


def _cross_ref_bank_statement(
    data: dict[str, Any],
    field_confidences: dict[str, Any]
) -> dict[str, Any]:
    """
    Cross-reference validation for bank statements.

    Checks if beginning_balance + transactions = ending_balance
    """
    header = data.get("header", {})
    beginning = header.get("beginning_balance")
    ending = header.get("ending_balance")
    transactions = data.get("transactions", [])

    if beginning is None or ending is None:
        return field_confidences

    try:
        beginning_val = float(beginning)
        ending_val = float(ending)
        transactions_sum = sum(
            float(t.get("amount", 0) or 0)
            for t in transactions
        )

        # Check if balances reconcile (within 1 cent)
        if abs((beginning_val + transactions_sum) - ending_val) < 0.01:
            # Boost balance field confidences
            for field in ["header.beginning_balance", "header.ending_balance"]:
                if field in field_confidences:
                    current = field_confidences[field]
                    if isinstance(current, (int, float)):
                        field_confidences[field] = min(current + 0.10, 1.0)

            # Also boost transaction amounts
            if "transactions" in field_confidences:
                trans_confs = field_confidences["transactions"]
                if isinstance(trans_confs, list):
                    for i, t_conf in enumerate(trans_confs):
                        if isinstance(t_conf, dict) and "amount" in t_conf:
                            t_conf["amount"] = min(t_conf["amount"] + 0.10, 1.0)

    except (TypeError, ValueError):
        pass

    return field_confidences


def _cross_ref_w2(
    data: dict[str, Any],
    field_confidences: dict[str, Any]
) -> dict[str, Any]:
    """
    Cross-reference validation for W2 forms.

    Checks if SS wages >= SS tax (6.2% relationship)
    Checks if Medicare wages >= Medicare tax (1.45% relationship)
    """
    # TODO: Assumed basic cross-reference checks. Revisit if more complex validation needed.
    wages = data.get("wages", {}) or data.get("boxes", {})

    try:
        ss_wages = float(wages.get("box3_ss_wages") or 0)
        ss_tax = float(wages.get("box4_ss_tax") or wages.get("box4_ss_withheld") or 0)

        # SS tax should be ~6.2% of wages
        if ss_wages > 0 and abs(ss_tax - (ss_wages * 0.062)) < 10:
            for field in ["boxes.box3_ss_wages", "boxes.box4_ss_withheld"]:
                if field in field_confidences:
                    current = field_confidences[field]
                    if isinstance(current, (int, float)):
                        field_confidences[field] = min(current + 0.10, 1.0)

        medicare_wages = float(wages.get("box5_medicare_wages") or 0)
        medicare_tax = float(
            wages.get("box6_medicare_tax") or
            wages.get("box6_medicare_withheld") or 0
        )

        # Medicare tax should be ~1.45% of wages
        if medicare_wages > 0 and abs(medicare_tax - (medicare_wages * 0.0145)) < 10:
            for field in ["boxes.box5_medicare_wages", "boxes.box6_medicare_withheld"]:
                if field in field_confidences:
                    current = field_confidences[field]
                    if isinstance(current, (int, float)):
                        field_confidences[field] = min(current + 0.10, 1.0)

    except (TypeError, ValueError):
        pass

    return field_confidences


def calculate_overall_confidence(
    field_confidences: dict[str, Any],
    critical_fields: list[str]
) -> float:
    """
    Calculate overall confidence per contract specification.

    Formula: min(mean(critical_field_confidences), min(critical_field_confidences) + 0.1)

    Args:
        field_confidences: Dict of field name -> confidence score
        critical_fields: List of critical field names for this doc type

    Returns:
        Overall confidence score (0.0 - 1.0)
    """
    critical_scores = []

    for field in critical_fields:
        if "[*]" in field:
            # Handle array fields like "transactions[*].amount"
            base_field = field.split("[*]")[0]  # "transactions"
            sub_field = field.split(".")[-1]     # "amount"

            if base_field in field_confidences:
                arr_confs = field_confidences[base_field]
                if isinstance(arr_confs, list):
                    for item in arr_confs:
                        if isinstance(item, dict) and sub_field in item:
                            critical_scores.append(item[sub_field])
        elif field in field_confidences:
            score = field_confidences[field]
            if isinstance(score, (int, float)):
                critical_scores.append(score)

    if not critical_scores:
        return 0.5  # Default when no critical fields found

    mean_critical = sum(critical_scores) / len(critical_scores)
    min_critical = min(critical_scores)

    return min(mean_critical, min_critical + 0.1)


def get_critical_fields(doc_type: str) -> list[str]:
    """
    Get critical fields for a document type.

    Args:
        doc_type: Document type

    Returns:
        List of critical field names
    """
    return CRITICAL_FIELDS.get(doc_type, [])
