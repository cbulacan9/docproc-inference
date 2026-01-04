"""
Transform raw dots.ocr output to document-specific schemas.

dots.ocr provides generic layout elements with bounding boxes and text.
These transformers convert that to structured fields matching our
document type schemas (W2, bank_statement, etc.).
"""

import re
from typing import Any


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


def transform_bank_statement(elements: list, raw_text: str) -> dict:
    """
    Transform dots.ocr output to bank statement schema.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Structured data matching the bank_statement schema
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

    # Extract common bank names
    bank_patterns = [
        r'(Chase|Bank of America|Wells Fargo|Citi|Capital One|PNC|TD Bank|US Bank)',
        r'([A-Z][a-z]+ (?:Bank|Credit Union|FCU))',
    ]
    for pattern in bank_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            result["header"]["bank_name"] = match.group(1)
            break

    # Extract account number (masked)
    acct_match = re.search(
        r'(?:Account|Acct)[:\s#]*(\*{2,}\d{4}|\d{4})',
        raw_text,
        re.IGNORECASE
    )
    if acct_match:
        result["header"]["account_number"] = f"****{acct_match.group(1)[-4:]}"

    # Extract dates for statement period
    dates = extract_dates(raw_text)
    if len(dates) >= 2:
        result["header"]["statement_period"] = f"{dates[0]} - {dates[1]}"

    # Look for beginning/ending balance patterns
    for element in elements:
        text = element.get("text", "").lower()
        if "beginning" in text or "opening" in text:
            amts = extract_amounts(element.get("text", ""))
            if amts:
                result["header"]["beginning_balance"] = amts[-1]["value"]
        elif "ending" in text or "closing" in text:
            amts = extract_amounts(element.get("text", ""))
            if amts:
                result["header"]["ending_balance"] = amts[-1]["value"]

    # Extract transactions (look for table-like patterns)
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
        except ValueError:
            continue

    return result


def transform_w2(elements: list, raw_text: str) -> dict:
    """
    Transform dots.ocr output to W2 form schema.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Structured data matching the W2 schema
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
        "wages": {
            "box1_wages": None,
            "box2_federal_tax": None,
            "box3_ss_wages": None,
            "box4_ss_tax": None,
            "box5_medicare_wages": None,
            "box6_medicare_tax": None
        },
        "tax_year": None
    }

    # Extract SSN
    result["employee"]["ssn"] = extract_ssn(raw_text)

    # Extract EIN
    result["employer"]["ein"] = extract_ein(raw_text)

    # Extract tax year (look for 4-digit year near "W-2" or "Wage")
    year_match = re.search(r'(?:W-2|Wage)[^\d]*(\d{4})', raw_text, re.IGNORECASE)
    if year_match:
        result["tax_year"] = year_match.group(1)
    else:
        # Try to find any 4-digit year in 20xx range
        year_match = re.search(r'\b(20\d{2})\b', raw_text)
        if year_match:
            result["tax_year"] = year_match.group(1)

    # Extract wage boxes (look for box labels)
    box_patterns = {
        "box1_wages": r'(?:Box\s*1|Wages,?\s*tips)[^\d]*\$?([\d,]+\.?\d*)',
        "box2_federal_tax": r'(?:Box\s*2|Federal\s*income\s*tax)[^\d]*\$?([\d,]+\.?\d*)',
        "box3_ss_wages": r'(?:Box\s*3|Social\s*security\s*wages)[^\d]*\$?([\d,]+\.?\d*)',
        "box4_ss_tax": r'(?:Box\s*4|Social\s*security\s*tax)[^\d]*\$?([\d,]+\.?\d*)',
        "box5_medicare_wages": r'(?:Box\s*5|Medicare\s*wages)[^\d]*\$?([\d,]+\.?\d*)',
        "box6_medicare_tax": r'(?:Box\s*6|Medicare\s*tax)[^\d]*\$?([\d,]+\.?\d*)',
    }

    for field, pattern in box_patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                result["wages"][field] = float(match.group(1).replace(',', ''))
            except ValueError:
                continue

    return result


def transform_1099(elements: list, raw_text: str) -> dict:
    """
    Transform dots.ocr output to 1099 form schema.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Structured data matching the 1099 schema
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
        "amounts": {},
        "tax_year": None,
        "form_type": None
    }

    # Detect 1099 type
    type_match = re.search(r'1099-?(INT|DIV|MISC|NEC|R|G|K)', raw_text, re.IGNORECASE)
    if type_match:
        result["form_type"] = f"1099-{type_match.group(1).upper()}"

    # Extract TINs
    result["recipient"]["tin"] = extract_ssn(raw_text)

    # Extract tax year
    year_match = re.search(r'\b(20\d{2})\b', raw_text)
    if year_match:
        result["tax_year"] = year_match.group(1)

    # Extract amounts from the document
    all_amounts = extract_amounts(raw_text)
    if all_amounts:
        # Sort by value descending to get most significant amounts
        sorted_amounts = sorted(all_amounts, key=lambda x: x["value"], reverse=True)
        for i, amt in enumerate(sorted_amounts[:5]):
            result["amounts"][f"amount_{i+1}"] = amt["value"]

    return result


def transform_generic(elements: list, raw_text: str) -> dict:
    """
    Transform dots.ocr output to generic document schema.

    Used for document types without specific transformers.

    Args:
        elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Generic structured data with extracted fields
    """
    result = {
        "amounts": extract_amounts(raw_text),
        "dates": extract_dates(raw_text),
        "text_blocks": [],
        "tables": []
    }

    # Group elements by category
    for element in elements:
        category = element.get("category", "Text")
        if category == "Table":
            result["tables"].append({
                "bbox": element.get("bbox"),
                "text": element.get("text", "")
            })
        else:
            result["text_blocks"].append({
                "category": category,
                "text": element.get("text", ""),
                "confidence": element.get("confidence", 0.9)
            })

    return result


def transform_to_document_schema(
    doc_type: str,
    layout_elements: list,
    raw_text: str
) -> dict:
    """
    Transform raw dots.ocr output to document-specific schema.

    Dispatcher function that routes to the appropriate transformer
    based on document type.

    Args:
        doc_type: Document type (W2, bank_statement, 1099-*, etc.)
        layout_elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages

    Returns:
        Structured data matching the document type schema
    """
    transformers = {
        "bank_statement": transform_bank_statement,
        "W2": transform_w2,
        "1099-INT": transform_1099,
        "1099-DIV": transform_1099,
        "1099-MISC": transform_1099,
        "1099-NEC": transform_1099,
        "1099-R": transform_1099,
    }

    transformer = transformers.get(doc_type, transform_generic)
    return transformer(layout_elements, raw_text)
