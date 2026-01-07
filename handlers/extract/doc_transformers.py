"""
Transform raw dots.ocr output to document-specific schemas.

dots.ocr provides generic layout elements with bounding boxes and text.
These transformers convert that to structured fields matching our
document type schemas (W2, bank_statement, etc.).

Each transformer returns a tuple of (data, field_confidences) where:
- data: Structured document fields
- field_confidences: Dict with dot-notation keys mapping to confidence scores
"""

import re
from typing import Any

from confidence import (
    apply_cross_reference_boost,
    apply_format_boost,
    calculate_overall_confidence,
    get_base_confidence,
    get_critical_fields,
)


def extract_amounts(text: str) -> list[dict]:
    """
    Extract monetary amounts from text.

    Args:
        text: Raw text to search

    Returns:
        List of dicts with raw, value, and position keys
    """
    amounts = []
    # Match patterns like $1,234.56 or 1234.56
    pattern = r'\$?([\d,]+\.?\d*)'
    for match in re.finditer(pattern, text):
        try:
            value_str = match.group(1).replace(',', '')
            if not value_str or value_str == '.':
                continue
            value = float(value_str)
            amounts.append({
                "raw": match.group(0),
                "value": value,
                "position": match.start()
            })
        except ValueError:
            continue
    return amounts


def extract_dates(text: str) -> list[str]:
    """
    Extract dates from text.

    Args:
        text: Raw text to search

    Returns:
        List of date strings found
    """
    dates = []
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or M/D/YY
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
        r'\d{4}-\d{2}-\d{2}',         # YYYY-MM-DD (ISO)
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
    ]
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return dates


def extract_ssn(text: str) -> str | None:
    """
    Extract Social Security Number from text.

    Args:
        text: Raw text to search

    Returns:
        Masked SSN (XXX-XX-1234) or None if not found
    """
    # Match full SSN or last 4 digits
    patterns = [
        r'\d{3}-\d{2}-(\d{4})',  # Full SSN format
        r'XXX-XX-(\d{4})',       # Already masked
        r'\*{3}-\*{2}-(\d{4})',  # Star masked
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"XXX-XX-{match.group(1)}"
    return None


def extract_ein(text: str) -> str | None:
    """
    Extract Employer Identification Number from text.

    Args:
        text: Raw text to search

    Returns:
        EIN string or None if not found
    """
    match = re.search(r'\d{2}-\d{7}', text)
    if match:
        return match.group(0)
    return None


def _get_element_confidence(
    elements: list,
    search_text: str,
    default: float = 0.85
) -> float:
    """
    Find confidence for an element containing specific text.

    Args:
        elements: List of layout elements
        search_text: Text to search for (case-insensitive)
        default: Default confidence if not found

    Returns:
        Confidence score from matching element
    """
    search_lower = search_text.lower()
    for element in elements:
        el_text = element.get("text", "").lower()
        if search_lower in el_text:
            return get_base_confidence(element)
    return default


def transform_bank_statement(
    elements: list,
    raw_text: str
) -> tuple[dict, dict[str, Any]]:
    """
    Transform dots.ocr output to bank statement schema.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Tuple of (structured_data, field_confidences)
    """
    result = {
        "header": {
            "bank_name": None,
            "account_number": None,
            "account_type": None,
            "statement_period": None,
            "beginning_balance": None,
            "ending_balance": None
        },
        "transactions": [],
        "summary": {
            "total_credits": None,
            "total_debits": None,
            "net_change": None
        }
    }

    confidences: dict[str, Any] = {}

    # Extract common bank names
    bank_patterns = [
        r'(Chase|Bank of America|Wells Fargo|Citi|Capital One|PNC|TD Bank|US Bank)',
        r'([A-Z][a-z]+ (?:Bank|Credit Union|FCU))',
    ]
    for pattern in bank_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            result["header"]["bank_name"] = match.group(1)
            base_conf = _get_element_confidence(elements, match.group(1))
            confidences["header.bank_name"] = apply_format_boost(
                match.group(1), "header.bank_name", base_conf
            )
            break

    if result["header"]["bank_name"] is None:
        confidences["header.bank_name"] = 0.0

    # Extract account number (masked)
    acct_match = re.search(
        r'(?:Account|Acct)[:\s#]*(\*{2,}\d{4}|\d{4})',
        raw_text,
        re.IGNORECASE
    )
    if acct_match:
        result["header"]["account_number"] = f"****{acct_match.group(1)[-4:]}"
        base_conf = _get_element_confidence(elements, acct_match.group(0))
        confidences["header.account_number"] = apply_format_boost(
            result["header"]["account_number"], "header.account_number", base_conf
        )
    else:
        confidences["header.account_number"] = 0.0

    # Account type - often not explicitly stated
    # TODO: Assumed account_type detection is optional. Revisit if needed.
    confidences["header.account_type"] = 0.0 if result["header"]["account_type"] is None else 0.95

    # Extract dates for statement period
    dates = extract_dates(raw_text)
    if len(dates) >= 2:
        result["header"]["statement_period"] = f"{dates[0]} - {dates[1]}"
        base_conf = _get_element_confidence(elements, dates[0], 0.90)
        confidences["header.statement_period"] = apply_format_boost(
            dates[0], "header.statement_period", base_conf
        )
    else:
        confidences["header.statement_period"] = 0.0

    # Look for beginning/ending balance patterns
    for element in elements:
        text = element.get("text", "").lower()
        el_conf = get_base_confidence(element)

        if "beginning" in text or "opening" in text:
            amts = extract_amounts(element.get("text", ""))
            if amts:
                result["header"]["beginning_balance"] = amts[-1]["value"]
                confidences["header.beginning_balance"] = apply_format_boost(
                    amts[-1]["raw"], "header.beginning_balance", el_conf
                )
        elif "ending" in text or "closing" in text:
            amts = extract_amounts(element.get("text", ""))
            if amts:
                result["header"]["ending_balance"] = amts[-1]["value"]
                confidences["header.ending_balance"] = apply_format_boost(
                    amts[-1]["raw"], "header.ending_balance", el_conf
                )

    # Set 0.0 confidence for missing balance fields
    if "header.beginning_balance" not in confidences:
        confidences["header.beginning_balance"] = 0.0
    if "header.ending_balance" not in confidences:
        confidences["header.ending_balance"] = 0.0

    # Extract transactions (look for table-like patterns)
    transaction_confidences = []
    transaction_pattern = r'(\d{1,2}/\d{1,2})\s+(.+?)\s+(-?\$?[\d,]+\.?\d*)\s*$'
    for match in re.finditer(transaction_pattern, raw_text, re.MULTILINE):
        date, desc, amount = match.groups()
        try:
            amt_value = float(amount.replace('$', '').replace(',', ''))
            result["transactions"].append({
                "date": date,
                "description": desc.strip(),
                "amount": amt_value
            })

            # Get base confidence from nearest element
            base_conf = _get_element_confidence(elements, date, 0.85)

            transaction_confidences.append({
                "date": apply_format_boost(date, "transactions.date", base_conf),
                "description": base_conf,  # No format boost for description
                "amount": apply_format_boost(amount, "transactions.amount", base_conf)
            })
        except ValueError:
            continue

    confidences["transactions"] = transaction_confidences

    # Summary fields - often derived or not present
    # TODO: Assumed summary extraction is optional. Revisit if explicit extraction needed.
    confidences["summary.total_credits"] = 0.0
    confidences["summary.total_debits"] = 0.0
    confidences["summary.net_change"] = 0.0

    return result, confidences


def transform_w2(
    elements: list,
    raw_text: str
) -> tuple[dict, dict[str, Any]]:
    """
    Transform dots.ocr output to W2 form schema.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Tuple of (structured_data, field_confidences)
    """
    result = {
        "employee": {
            "name": None,
            "ssn": None,
            "address": None
        },
        "employer": {
            "name": None,
            "ein": None,
            "address": None
        },
        "boxes": {
            "box1_wages": None,
            "box2_federal_withheld": None,
            "box3_ss_wages": None,
            "box4_ss_withheld": None,
            "box5_medicare_wages": None,
            "box6_medicare_withheld": None
        },
        "tax_year": None
    }

    confidences: dict[str, Any] = {}

    # Extract SSN
    ssn = extract_ssn(raw_text)
    result["employee"]["ssn"] = ssn
    if ssn:
        base_conf = _get_element_confidence(elements, ssn[-4:], 0.90)
        confidences["employee.ssn"] = apply_format_boost(ssn, "employee.ssn", base_conf)
    else:
        confidences["employee.ssn"] = 0.0

    # Extract EIN
    ein = extract_ein(raw_text)
    result["employer"]["ein"] = ein
    if ein:
        base_conf = _get_element_confidence(elements, ein, 0.90)
        confidences["employer.ein"] = apply_format_boost(ein, "employer.ein", base_conf)
    else:
        confidences["employer.ein"] = 0.0

    # Employee/Employer names - set to 0.0 if not found
    confidences["employee.name"] = 0.0
    confidences["employee.address"] = 0.0
    confidences["employer.name"] = 0.0
    confidences["employer.address"] = 0.0

    # Extract tax year (look for 4-digit year near "W-2" or "Wage")
    year_match = re.search(r'(?:W-2|Wage)[^\d]*(\d{4})', raw_text, re.IGNORECASE)
    if year_match:
        result["tax_year"] = year_match.group(1)
        base_conf = _get_element_confidence(elements, year_match.group(1), 0.95)
        confidences["tax_year"] = base_conf
    else:
        # Try to find any 4-digit year in 20xx range
        year_match = re.search(r'\b(20\d{2})\b', raw_text)
        if year_match:
            result["tax_year"] = year_match.group(1)
            confidences["tax_year"] = 0.80  # Lower confidence for generic year match
        else:
            confidences["tax_year"] = 0.0

    # Extract wage boxes (look for box labels)
    box_patterns = {
        "box1_wages": r'(?:Box\s*1|Wages,?\s*tips)[^\d]*\$?([\d,]+\.?\d*)',
        "box2_federal_withheld": r'(?:Box\s*2|Federal\s*income\s*tax)[^\d]*\$?([\d,]+\.?\d*)',
        "box3_ss_wages": r'(?:Box\s*3|Social\s*security\s*wages)[^\d]*\$?([\d,]+\.?\d*)',
        "box4_ss_withheld": r'(?:Box\s*4|Social\s*security\s*tax)[^\d]*\$?([\d,]+\.?\d*)',
        "box5_medicare_wages": r'(?:Box\s*5|Medicare\s*wages)[^\d]*\$?([\d,]+\.?\d*)',
        "box6_medicare_withheld": r'(?:Box\s*6|Medicare\s*tax)[^\d]*\$?([\d,]+\.?\d*)',
    }

    for field, pattern in box_patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(',', ''))
                result["boxes"][field] = value
                base_conf = _get_element_confidence(elements, match.group(0)[:20], 0.90)
                confidences[f"boxes.{field}"] = apply_format_boost(
                    match.group(1), f"boxes.{field}", base_conf
                )
            except ValueError:
                confidences[f"boxes.{field}"] = 0.0
        else:
            confidences[f"boxes.{field}"] = 0.0

    return result, confidences


def transform_1099(
    elements: list,
    raw_text: str
) -> tuple[dict, dict[str, Any]]:
    """
    Transform dots.ocr output to 1099 form schema.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Tuple of (structured_data, field_confidences)
    """
    result = {
        "recipient": {
            "name": None,
            "tin": None,
            "address": None
        },
        "payer": {
            "name": None,
            "tin": None,
            "address": None
        },
        "boxes": {},
        "tax_year": None,
        "form_type": None
    }

    confidences: dict[str, Any] = {}

    # Detect 1099 type
    type_match = re.search(r'1099-?(INT|DIV|MISC|NEC|R|G|K)', raw_text, re.IGNORECASE)
    if type_match:
        result["form_type"] = f"1099-{type_match.group(1).upper()}"
        confidences["form_type"] = 0.95
    else:
        confidences["form_type"] = 0.0

    # Extract recipient TIN
    tin = extract_ssn(raw_text)
    result["recipient"]["tin"] = tin
    if tin:
        base_conf = _get_element_confidence(elements, tin[-4:], 0.90)
        confidences["recipient.tin"] = apply_format_boost(tin, "recipient.tin", base_conf)
    else:
        confidences["recipient.tin"] = 0.0

    # Set default confidences for name/address fields
    confidences["recipient.name"] = 0.0
    confidences["recipient.address"] = 0.0
    confidences["payer.name"] = 0.0
    confidences["payer.tin"] = 0.0
    confidences["payer.address"] = 0.0

    # Extract tax year
    year_match = re.search(r'\b(20\d{2})\b', raw_text)
    if year_match:
        result["tax_year"] = year_match.group(1)
        base_conf = _get_element_confidence(elements, year_match.group(1), 0.90)
        confidences["tax_year"] = base_conf
    else:
        confidences["tax_year"] = 0.0

    # Extract amounts from the document
    all_amounts = extract_amounts(raw_text)
    if all_amounts:
        # Sort by value descending to get most significant amounts
        sorted_amounts = sorted(all_amounts, key=lambda x: x["value"], reverse=True)
        for i, amt in enumerate(sorted_amounts[:5]):
            box_key = f"box{i+1}"
            result["boxes"][box_key] = amt["value"]
            base_conf = _get_element_confidence(elements, amt["raw"], 0.85)
            confidences[f"boxes.{box_key}"] = apply_format_boost(
                amt["raw"], f"boxes.{box_key}", base_conf
            )

    return result, confidences


def transform_generic(
    elements: list,
    raw_text: str
) -> tuple[dict, dict[str, Any]]:
    """
    Transform dots.ocr output to generic document schema.

    Used for document types without specific transformers.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Tuple of (structured_data, field_confidences)
    """
    amounts = extract_amounts(raw_text)
    dates = extract_dates(raw_text)

    result = {
        "amounts": amounts,
        "dates": dates,
        "text_blocks": [],
        "tables": []
    }

    confidences: dict[str, Any] = {
        "amounts": [],
        "dates": [],
        "text_blocks": [],
        "tables": []
    }

    # Group elements by category
    for element in elements:
        category = element.get("category", "Text")
        el_conf = get_base_confidence(element)

        if category == "Table":
            result["tables"].append({
                "bbox": element.get("bbox"),
                "text": element.get("text", "")
            })
            confidences["tables"].append({"text": el_conf})
        else:
            result["text_blocks"].append({
                "category": category,
                "text": element.get("text", ""),
                "confidence": el_conf
            })
            confidences["text_blocks"].append({"text": el_conf})

    # Add confidence for extracted amounts
    for amt in amounts:
        confidences["amounts"].append({"value": 0.85})

    # Add confidence for extracted dates
    for _ in dates:
        confidences["dates"].append({"value": 0.85})

    return result, confidences


# Transformer registry
TRANSFORMERS = {
    "bank_statement": transform_bank_statement,
    "W2": transform_w2,
    "1099-INT": transform_1099,
    "1099-DIV": transform_1099,
    "1099-MISC": transform_1099,
    "1099-NEC": transform_1099,
    "1099-R": transform_1099,
}


def transform_to_document_schema(
    doc_type: str,
    layout_elements: list,
    raw_text: str
) -> tuple[dict, dict[str, Any]]:
    """
    Transform raw dots.ocr output to document-specific schema.

    Dispatcher function that routes to the appropriate transformer
    based on document type. Returns both structured data and
    per-field confidence scores.

    Args:
        doc_type: Document type (W2, bank_statement, 1099-*, etc.)
        layout_elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Tuple of (data_dict, confidence_dict) where confidence_dict
        contains {"overall": float, "fields": dict}
    """
    transformer = TRANSFORMERS.get(doc_type, transform_generic)
    data, field_confidences = transformer(layout_elements, raw_text)

    # Apply cross-reference boost
    field_confidences = apply_cross_reference_boost(data, field_confidences, doc_type)

    # Calculate overall confidence
    critical_fields = get_critical_fields(doc_type)
    overall = calculate_overall_confidence(field_confidences, critical_fields)

    return data, {"overall": overall, "fields": field_confidences}
