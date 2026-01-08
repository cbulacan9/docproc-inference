"""
Confidence derivation for Chandra OCR extraction.

Since Chandra does not provide OCR-level confidence scores, confidence
is derived from:
1. Format validation - values matching expected patterns
2. Structural context - where/how values were found
3. Cross-reference validation - values that cross-validate

Base confidence for Chandra is 0.80 (vs 0.85 for dots.ocr with element scores).
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
    "1099-INT": ["recipient.tin", "boxes.box1"],
    "1099-DIV": ["recipient.tin", "boxes.box1"],
    "1099-MISC": ["recipient.tin", "boxes.box1"],
    "1099-NEC": ["recipient.tin", "boxes.box1"],
    "1099-R": ["recipient.tin", "boxes.box1"],
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


def derive_confidence_from_format(
    value: Any,
    field_name: str,
    found_in_table: bool = False,
    found_as_kv: bool = False
) -> float:
    """
    Derive confidence based on format validation and structural context.

    Base confidence: 0.80 (Chandra default without OCR scores)
    + 0.10 if format matches expected pattern
    + 0.05 if value is non-empty and reasonable length
    + 0.05 if found in table (structured data)
    + 0.03 if found as key-value pair (labeled)

    Args:
        value: Field value to validate
        field_name: Dot-notation field name
        found_in_table: Whether value was extracted from a table
        found_as_kv: Whether value was extracted as a labeled key-value pair

    Returns:
        Derived confidence score (capped at 0.99)
    """
    if value is None:
        return 0.0

    base = 0.80

    # Get field type from mapping
    field_type = FIELD_TYPE_MAPPING.get(field_name)

    # Also check for transaction subfields
    if not field_type and "." in field_name:
        parts = field_name.split(".")
        generic_key = f"{parts[0]}.{parts[-1]}"
        field_type = FIELD_TYPE_MAPPING.get(generic_key)

    # Format match boost (+0.10)
    if field_type:
        pattern = FORMAT_PATTERNS.get(field_type)
        if pattern:
            str_value = str(value).strip()
            if re.match(pattern, str_value):
                base += 0.10

    # Non-empty boost (+0.05)
    str_value = str(value).strip() if value else ""
    if len(str_value) > 0:
        base += 0.05

    # Structural boosts
    if found_in_table:
        base += 0.05
    if found_as_kv:
        base += 0.03

    # Cap at 0.99 since we don't have OCR-level confidence
    return min(base, 0.99)


def derive_structural_confidence(
    parsed: dict,
    field_name: str,
    value: Any
) -> float:
    """
    Derive confidence based on structural context in parsed output.

    Higher confidence if:
    - Field found in expected section
    - Field extracted from table
    - Field has explicit label (key-value pair)

    Args:
        parsed: Parsed Chandra output
        field_name: Dot-notation field name
        value: Extracted value

    Returns:
        Derived confidence score
    """
    found_in_table = False
    found_as_kv = False

    str_value = str(value).strip().lower() if value else ""

    # Check if value appears in any table
    for table in parsed.get("tables", []):
        for row in table.get("rows", []):
            for cell in row:
                if str_value and str_value in str(cell).lower():
                    found_in_table = True
                    break

    # Check if value appears in key-value pairs
    for kv in parsed.get("key_value_pairs", []):
        if str_value and str_value in str(kv.get("value", "")).lower():
            found_as_kv = True
            break

    return derive_confidence_from_format(
        value, field_name, found_in_table, found_as_kv
    )


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
                        field_confidences[field] = min(current + 0.10, 0.99)

            # Also boost transaction amounts
            if "transactions" in field_confidences:
                trans_confs = field_confidences["transactions"]
                if isinstance(trans_confs, list):
                    for t_conf in trans_confs:
                        if isinstance(t_conf, dict) and "amount" in t_conf:
                            t_conf["amount"] = min(t_conf["amount"] + 0.10, 0.99)

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
    boxes = data.get("boxes", {})

    try:
        ss_wages = float(boxes.get("box3_ss_wages") or 0)
        ss_tax = float(boxes.get("box4_ss_withheld") or 0)

        # SS tax should be ~6.2% of wages
        if ss_wages > 0 and abs(ss_tax - (ss_wages * 0.062)) < 10:
            for field in ["boxes.box3_ss_wages", "boxes.box4_ss_withheld"]:
                if field in field_confidences:
                    current = field_confidences[field]
                    if isinstance(current, (int, float)):
                        field_confidences[field] = min(current + 0.10, 0.99)

        medicare_wages = float(boxes.get("box5_medicare_wages") or 0)
        medicare_tax = float(boxes.get("box6_medicare_withheld") or 0)

        # Medicare tax should be ~1.45% of wages
        if medicare_wages > 0 and abs(medicare_tax - (medicare_wages * 0.0145)) < 10:
            for field in ["boxes.box5_medicare_wages", "boxes.box6_medicare_withheld"]:
                if field in field_confidences:
                    current = field_confidences[field]
                    if isinstance(current, (int, float)):
                        field_confidences[field] = min(current + 0.10, 0.99)

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
            base_field = field.split("[*]")[0]
            sub_field = field.split(".")[-1]

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
