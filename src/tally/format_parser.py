"""
Format string parser for custom CSV column mappings.

Parses format strings like: {date:%m/%d/%Y}, {description}, {_}, {amount}
Position in the string implies column index.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FormatSpec:
    """Parsed format specification for CSV parsing."""
    date_column: int
    date_format: str
    description_column: int
    amount_column: int
    location_column: Optional[int] = None
    has_header: bool = True
    source_name: Optional[str] = None  # Optional override for transaction source
    negate_amount: bool = False  # If True, flip the sign of amounts (for credit cards)


def parse_format_string(format_str: str) -> FormatSpec:
    """
    Parse a format string into a FormatSpec.

    Format string syntax:
        {field}           - Map this column to a field
        {field:format}    - Field with format specifier (e.g., date format)
        {-field}          - Negate the value (for amount: flip sign)
        {_}               - Skip this column

    Required fields: date, description, amount
    Optional fields: location

    Examples:
        "{date:%m/%d/%Y}, {description}, {amount}"
        "{date:%Y-%m-%d}, {_}, {description}, {location}, {amount}"
        "{date:%m/%d/%Y}, {description}, {-amount}"  # Negate amounts (credit cards)

    Args:
        format_str: The format string to parse

    Returns:
        FormatSpec with column mappings

    Raises:
        ValueError: If format string is invalid or missing required fields
    """
    # Pattern to match {field}, {-field}, or {field:format}
    field_pattern = re.compile(r'\{(-?)(\w+)(?::([^}]+))?\}')

    # Split by comma and parse each column
    parts = [p.strip() for p in format_str.split(',')]

    if not parts:
        raise ValueError("Empty format string")

    field_positions = {}
    date_format = '%m/%d/%Y'  # Default
    negate_amount = False

    for idx, part in enumerate(parts):
        match = field_pattern.match(part)
        if not match:
            raise ValueError(f"Invalid format at column {idx}: '{part}'. Expected {{field}} or {{field:format}}")

        negate_prefix = match.group(1)  # '-' or ''
        field_name = match.group(2).lower()
        format_spec = match.group(3)  # May be None

        # Skip placeholder columns
        if field_name == '_':
            continue

        # Validate field name
        valid_fields = {'date', 'description', 'amount', 'location'}
        if field_name not in valid_fields:
            raise ValueError(f"Unknown field '{field_name}' at column {idx}. Valid fields: {valid_fields}")

        # Check for duplicates
        if field_name in field_positions:
            raise ValueError(f"Duplicate field '{field_name}' at column {idx}")

        field_positions[field_name] = idx

        # Capture date format if specified
        if field_name == 'date' and format_spec:
            date_format = format_spec

        # Capture negation for amount
        if field_name == 'amount' and negate_prefix == '-':
            negate_amount = True

    # Validate required fields
    required = {'date', 'description', 'amount'}
    missing = required - set(field_positions.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return FormatSpec(
        date_column=field_positions['date'],
        date_format=date_format,
        description_column=field_positions['description'],
        amount_column=field_positions['amount'],
        location_column=field_positions.get('location'),
        has_header=True,
        negate_amount=negate_amount
    )


# Predefined format shortcuts for backward compatibility
PREDEFINED_FORMATS = {
    # Standard AMEX CSV: Date,Description,Amount (with headers)
    'amex': None,  # Use legacy parser - handles header-based CSV
    # BOA text format - not a standard CSV, needs special parser
    'boa': None,   # Use legacy parser - regex-based line parsing
}


def get_predefined_format(source_type: str) -> Optional[str]:
    """
    Get the format string for a predefined source type.

    Returns None if the type requires a special parser (not generic CSV).
    """
    return PREDEFINED_FORMATS.get(source_type.lower())


def is_special_parser_type(source_type: str) -> bool:
    """Check if a source type requires a special (non-generic) parser."""
    return source_type.lower() in PREDEFINED_FORMATS
