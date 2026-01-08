"""
Transform Chandra parsed Markdown output to document-specific schemas.

Chandra outputs structured Markdown which is parsed into sections,
key-value pairs, tables, and lists. These transformers convert
that structure into our document type schemas with confidence scores.
"""

import re
from typing import Any

from confidence import (
    apply_cross_reference_boost,
    calculate_overall_confidence,
    derive_confidence_from_format,
    get_critical_fields,
)


def extract_currency(value: str) -> float | None:
    """
    Extract numeric value from currency string.

    Args:
        value: String like "$12,456.78" or "12456.78"

    Returns:
        Float value or None if parsing fails
    """
    if not value:
        return None

    try:
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,]', '', str(value).strip())
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def extract_account_number(value: str) -> str | None:
    """
    Extract and mask account number.

    Args:
        value: Raw account number string

    Returns:
        Masked account number (****XXXX format) or None
    """
    if not value:
        return None

    # Get last 4 digits
    digits = re.findall(r'\d', str(value))
    if len(digits) >= 4:
        return f"****{digits[-4]}{digits[-3]}{digits[-2]}{digits[-1]}"

    # Already masked
    if '****' in str(value) or '***' in str(value):
        return str(value).strip()

    return str(value).strip() if value else None


def extract_ssn(value: str) -> str | None:
    """
    Extract and mask SSN.

    Args:
        value: Raw SSN string

    Returns:
        Masked SSN (XXX-XX-XXXX format) or None
    """
    if not value:
        return None

    # Match full SSN or last 4 digits
    patterns = [
        r'\d{3}-\d{2}-(\d{4})',
        r'XXX-XX-(\d{4})',
        r'\*{3}-\*{2}-(\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, str(value))
        if match:
            return f"XXX-XX-{match.group(1)}"

    return None


def extract_ein(value: str) -> str | None:
    """
    Extract EIN.

    Args:
        value: Raw EIN string

    Returns:
        EIN string or None
    """
    if not value:
        return None

    match = re.search(r'\d{2}-\d{7}', str(value))
    return match.group(0) if match else None


def is_transaction_table(table: dict) -> bool:
    """
    Determine if a table contains transaction data.

    Args:
        table: Table dict with headers and rows

    Returns:
        True if table appears to be a transaction table
    """
    headers_lower = [h.lower() for h in table.get("headers", [])]

    # Look for common transaction column names
    transaction_indicators = ["date", "description", "amount", "debit", "credit", "balance"]
    matches = sum(1 for ind in transaction_indicators if any(ind in h for h in headers_lower))

    return matches >= 2


def parse_transaction_row(
    row: list[str],
    headers: list[str]
) -> tuple[dict, dict]:
    """
    Parse a table row into a transaction dict with confidence.

    Args:
        row: List of cell values
        headers: List of header names

    Returns:
        Tuple of (transaction_dict, confidence_dict)
    """
    txn = {
        "date": None,
        "description": None,
        "amount": None
    }
    conf = {}

    headers_lower = [h.lower() for h in headers]

    for i, (header, cell) in enumerate(zip(headers_lower, row)):
        cell_stripped = cell.strip()

        if "date" in header:
            txn["date"] = cell_stripped
            conf["date"] = derive_confidence_from_format(
                cell_stripped, "transactions.date", found_in_table=True
            )
        elif "desc" in header or "memo" in header or "detail" in header:
            txn["description"] = cell_stripped
            conf["description"] = derive_confidence_from_format(
                cell_stripped, "transactions.description", found_in_table=True
            )
        elif "amount" in header or "debit" in header or "credit" in header:
            amount = extract_currency(cell_stripped)
            if amount is not None:
                # Handle debit as negative
                if "debit" in header and amount > 0:
                    amount = -amount
                txn["amount"] = amount
                conf["amount"] = derive_confidence_from_format(
                    cell_stripped, "transactions.amount", found_in_table=True
                )

    return txn, conf


def transform_bank_statement(parsed: dict) -> tuple[dict, dict[str, Any]]:
    """
    Transform Chandra parsed output to bank statement schema.

    Args:
        parsed: Output from parse_chandra_output()

    Returns:
        Tuple of (data, field_confidences)
    """
    data = {
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

    # Extract bank name from first H1 header
    for section in parsed.get("sections", []):
        if section["level"] == 1:
            data["header"]["bank_name"] = section["title"]
            confidences["header.bank_name"] = 0.90  # H1 header = high confidence
            break

    if data["header"]["bank_name"] is None:
        confidences["header.bank_name"] = 0.0

    # Extract from key-value pairs
    for kv in parsed.get("key_value_pairs", []):
        key_lower = kv["key"].lower()
        value = kv["value"]

        if "account" in key_lower and "number" in key_lower:
            data["header"]["account_number"] = extract_account_number(value)
            confidences["header.account_number"] = derive_confidence_from_format(
                value, "header.account_number", found_as_kv=True
            )

        elif "statement" in key_lower and "period" in key_lower:
            data["header"]["statement_period"] = value
            confidences["header.statement_period"] = derive_confidence_from_format(
                value, "header.statement_period", found_as_kv=True
            )

        elif "beginning" in key_lower or "opening" in key_lower:
            data["header"]["beginning_balance"] = extract_currency(value)
            confidences["header.beginning_balance"] = derive_confidence_from_format(
                value, "header.beginning_balance", found_as_kv=True
            )

        elif "ending" in key_lower or "closing" in key_lower:
            data["header"]["ending_balance"] = extract_currency(value)
            confidences["header.ending_balance"] = derive_confidence_from_format(
                value, "header.ending_balance", found_as_kv=True
            )

        elif "total" in key_lower and "credit" in key_lower:
            data["summary"]["total_credits"] = extract_currency(value)
            confidences["summary.total_credits"] = derive_confidence_from_format(
                value, "summary.total_credits", found_as_kv=True
            )

        elif "total" in key_lower and "debit" in key_lower:
            data["summary"]["total_debits"] = extract_currency(value)
            confidences["summary.total_debits"] = derive_confidence_from_format(
                value, "summary.total_debits", found_as_kv=True
            )

        elif "net" in key_lower:
            data["summary"]["net_change"] = extract_currency(value)
            confidences["summary.net_change"] = derive_confidence_from_format(
                value, "summary.net_change", found_as_kv=True
            )

    # Set 0.0 confidence for missing fields
    for field in ["header.account_number", "header.statement_period",
                  "header.beginning_balance", "header.ending_balance"]:
        if field not in confidences:
            confidences[field] = 0.0

    # Account type not typically in Chandra output
    confidences["header.account_type"] = 0.0

    # Extract transactions from tables
    transaction_confidences = []
    for table in parsed.get("tables", []):
        if is_transaction_table(table):
            for row in table["rows"]:
                txn, txn_conf = parse_transaction_row(row, table["headers"])
                if txn["date"] or txn["amount"]:  # Only add if we got something
                    data["transactions"].append(txn)
                    transaction_confidences.append(txn_conf)

    confidences["transactions"] = transaction_confidences

    # Extract summary from lists if not found in key-value pairs
    for lst in parsed.get("lists", []):
        for item in lst.get("items", []):
            item_lower = item.lower()
            if "total credit" in item_lower and data["summary"]["total_credits"] is None:
                amount = extract_currency(item)
                if amount:
                    data["summary"]["total_credits"] = amount
                    confidences["summary.total_credits"] = 0.85
            elif "total debit" in item_lower and data["summary"]["total_debits"] is None:
                amount = extract_currency(item)
                if amount:
                    data["summary"]["total_debits"] = amount
                    confidences["summary.total_debits"] = 0.85
            elif "net" in item_lower and data["summary"]["net_change"] is None:
                amount = extract_currency(item)
                if amount:
                    data["summary"]["net_change"] = amount
                    confidences["summary.net_change"] = 0.85

    # Set 0.0 for missing summary fields
    for field in ["summary.total_credits", "summary.total_debits", "summary.net_change"]:
        if field not in confidences:
            confidences[field] = 0.0

    return data, confidences


def transform_w2(parsed: dict) -> tuple[dict, dict[str, Any]]:
    """
    Transform Chandra parsed output to W2 form schema.

    Args:
        parsed: Output from parse_chandra_output()

    Returns:
        Tuple of (data, field_confidences)
    """
    data = {
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

    # Extract from key-value pairs
    for kv in parsed.get("key_value_pairs", []):
        key_lower = kv["key"].lower()
        value = kv["value"]

        # Employee info
        if "employee" in key_lower and "ssn" in key_lower:
            data["employee"]["ssn"] = extract_ssn(value)
            confidences["employee.ssn"] = derive_confidence_from_format(
                value, "employee.ssn", found_as_kv=True
            )
        elif "employee" in key_lower and "name" in key_lower:
            data["employee"]["name"] = value
            confidences["employee.name"] = 0.88

        # Employer info
        elif "employer" in key_lower and "ein" in key_lower:
            data["employer"]["ein"] = extract_ein(value)
            confidences["employer.ein"] = derive_confidence_from_format(
                value, "employer.ein", found_as_kv=True
            )
        elif "employer" in key_lower and "name" in key_lower:
            data["employer"]["name"] = value
            confidences["employer.name"] = 0.88

        # Box values
        elif "box 1" in key_lower or "wages" in key_lower and "tip" in key_lower:
            data["boxes"]["box1_wages"] = extract_currency(value)
            confidences["boxes.box1_wages"] = derive_confidence_from_format(
                value, "boxes.box1_wages", found_as_kv=True
            )
        elif "box 2" in key_lower or "federal" in key_lower and "withheld" in key_lower:
            data["boxes"]["box2_federal_withheld"] = extract_currency(value)
            confidences["boxes.box2_federal_withheld"] = derive_confidence_from_format(
                value, "boxes.box2_federal_withheld", found_as_kv=True
            )
        elif "box 3" in key_lower or "social security wages" in key_lower:
            data["boxes"]["box3_ss_wages"] = extract_currency(value)
            confidences["boxes.box3_ss_wages"] = derive_confidence_from_format(
                value, "boxes.box3_ss_wages", found_as_kv=True
            )
        elif "box 4" in key_lower or "social security tax" in key_lower:
            data["boxes"]["box4_ss_withheld"] = extract_currency(value)
            confidences["boxes.box4_ss_withheld"] = derive_confidence_from_format(
                value, "boxes.box4_ss_withheld", found_as_kv=True
            )
        elif "box 5" in key_lower or "medicare wages" in key_lower:
            data["boxes"]["box5_medicare_wages"] = extract_currency(value)
            confidences["boxes.box5_medicare_wages"] = derive_confidence_from_format(
                value, "boxes.box5_medicare_wages", found_as_kv=True
            )
        elif "box 6" in key_lower or "medicare tax" in key_lower:
            data["boxes"]["box6_medicare_withheld"] = extract_currency(value)
            confidences["boxes.box6_medicare_withheld"] = derive_confidence_from_format(
                value, "boxes.box6_medicare_withheld", found_as_kv=True
            )
        elif "tax year" in key_lower or "year" in key_lower:
            year_match = re.search(r'20\d{2}', value)
            if year_match:
                data["tax_year"] = year_match.group(0)
                confidences["tax_year"] = 0.90

    # Set 0.0 for missing fields
    for field in ["employee.ssn", "employee.name", "employee.address",
                  "employer.ein", "employer.name", "employer.address",
                  "boxes.box1_wages", "boxes.box2_federal_withheld",
                  "boxes.box3_ss_wages", "boxes.box4_ss_withheld",
                  "boxes.box5_medicare_wages", "boxes.box6_medicare_withheld",
                  "tax_year"]:
        if field not in confidences:
            confidences[field] = 0.0

    return data, confidences


def transform_1099(parsed: dict) -> tuple[dict, dict[str, Any]]:
    """
    Transform Chandra parsed output to 1099 form schema.

    Args:
        parsed: Output from parse_chandra_output()

    Returns:
        Tuple of (data, field_confidences)
    """
    data = {
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

    # Detect form type from raw text
    raw_text = parsed.get("raw_text", "")
    type_match = re.search(r'1099-?(INT|DIV|MISC|NEC|R|G|K)', raw_text, re.IGNORECASE)
    if type_match:
        data["form_type"] = f"1099-{type_match.group(1).upper()}"
        confidences["form_type"] = 0.95
    else:
        confidences["form_type"] = 0.0

    # Extract from key-value pairs
    box_num = 1
    for kv in parsed.get("key_value_pairs", []):
        key_lower = kv["key"].lower()
        value = kv["value"]

        if "recipient" in key_lower and ("tin" in key_lower or "ssn" in key_lower):
            data["recipient"]["tin"] = extract_ssn(value)
            confidences["recipient.tin"] = derive_confidence_from_format(
                value, "recipient.tin", found_as_kv=True
            )
        elif "recipient" in key_lower and "name" in key_lower:
            data["recipient"]["name"] = value
            confidences["recipient.name"] = 0.88
        elif "payer" in key_lower and ("tin" in key_lower or "ein" in key_lower):
            data["payer"]["tin"] = extract_ein(value)
            confidences["payer.tin"] = derive_confidence_from_format(
                value, "payer.tin", found_as_kv=True
            )
        elif "payer" in key_lower and "name" in key_lower:
            data["payer"]["name"] = value
            confidences["payer.name"] = 0.88
        elif "box" in key_lower or any(x in key_lower for x in ["interest", "dividend", "income"]):
            amount = extract_currency(value)
            if amount:
                box_key = f"box{box_num}"
                data["boxes"][box_key] = amount
                confidences[f"boxes.{box_key}"] = derive_confidence_from_format(
                    value, f"boxes.{box_key}", found_as_kv=True
                )
                box_num += 1
        elif "tax year" in key_lower or "year" in key_lower:
            year_match = re.search(r'20\d{2}', value)
            if year_match:
                data["tax_year"] = year_match.group(0)
                confidences["tax_year"] = 0.90

    # Set 0.0 for missing fields
    for field in ["recipient.tin", "recipient.name", "recipient.address",
                  "payer.tin", "payer.name", "payer.address", "tax_year"]:
        if field not in confidences:
            confidences[field] = 0.0

    return data, confidences


def transform_generic(parsed: dict) -> tuple[dict, dict[str, Any]]:
    """
    Transform Chandra parsed output to generic document schema.

    Used for document types without specific transformers.

    Args:
        parsed: Output from parse_chandra_output()

    Returns:
        Tuple of (data, field_confidences)
    """
    data = {
        "sections": [],
        "key_value_pairs": [],
        "tables": [],
        "text_blocks": []
    }
    confidences: dict[str, Any] = {
        "sections": [],
        "key_value_pairs": [],
        "tables": [],
        "text_blocks": []
    }

    # Pass through sections
    for section in parsed.get("sections", []):
        data["sections"].append({
            "level": section["level"],
            "title": section["title"],
            "content": section.get("content", "")
        })
        confidences["sections"].append({"title": 0.90, "content": 0.85})

    # Pass through key-value pairs
    for kv in parsed.get("key_value_pairs", []):
        data["key_value_pairs"].append(kv)
        confidences["key_value_pairs"].append({"value": 0.85})

    # Pass through tables
    for table in parsed.get("tables", []):
        data["tables"].append(table)
        confidences["tables"].append({"data": 0.88})

    return data, confidences


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
    parsed: dict
) -> tuple[dict, dict[str, Any]]:
    """
    Transform Chandra parsed output to document-specific schema.

    Dispatcher function that routes to the appropriate transformer
    based on document type. Returns both structured data and
    per-field confidence scores.

    Args:
        doc_type: Document type (W2, bank_statement, 1099-*, etc.)
        parsed: Parsed Chandra output from parse_chandra_output()

    Returns:
        Tuple of (data_dict, confidence_dict) where confidence_dict
        contains {"overall": float, "fields": dict}
    """
    transformer = TRANSFORMERS.get(doc_type, transform_generic)
    data, field_confidences = transformer(parsed)

    # Apply cross-reference boost
    field_confidences = apply_cross_reference_boost(data, field_confidences, doc_type)

    # Calculate overall confidence
    critical_fields = get_critical_fields(doc_type)
    overall = calculate_overall_confidence(field_confidences, critical_fields)

    return data, {"overall": overall, "fields": field_confidences}
