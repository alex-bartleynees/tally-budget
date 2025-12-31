"""
Merchant normalization utilities for spending analysis.

This module provides functions to clean and categorize merchant descriptions
from credit card and bank statements.
"""

import csv
import os
import re
from datetime import date
from typing import Optional, List, Tuple

from .modifier_parser import (
    parse_pattern_with_modifiers,
    check_all_conditions,
    ParsedPattern,
    ModifierParseError,
)



def load_merchant_rules(csv_path):
    """Load user merchant categorization rules from CSV file.

    CSV format: Pattern,Merchant,Category,Subcategory[,Tags]

    Patterns support inline modifiers for amount/date matching:
        COSTCO[amount>200] - Match COSTCO transactions over $200
        BESTBUY[date=2025-01-15] - Match BESTBUY on specific date
        MERCHANT[amount:50-200][date:2025-01-01..2025-12-31] - Combined

    Tags are optional, pipe-separated: business|reimbursable

    Lines starting with # are treated as comments and skipped.
    Patterns are Python regular expressions matched against transaction descriptions.

    Returns list of tuples: (pattern, merchant_name, category, subcategory, parsed_pattern, tags)
    """
    if not os.path.exists(csv_path):
        return []  # No user rules file

    rules = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Filter out comment and empty lines before passing to DictReader
        lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            # Skip empty patterns
            pattern_str = row.get('Pattern', '').strip()
            if not pattern_str:
                continue

            # Parse pattern with inline modifiers
            try:
                parsed = parse_pattern_with_modifiers(pattern_str)
            except ModifierParseError:
                # Invalid modifier syntax - use pattern as-is without modifiers
                parsed = ParsedPattern(regex_pattern=pattern_str)

            # Parse tags (optional, pipe-separated)
            tags_str = row.get('Tags') or ''
            tags_str = tags_str.strip()
            tags = [t.strip() for t in tags_str.split('|') if t.strip()] if tags_str else []

            rules.append((
                parsed.regex_pattern,  # Pure regex for matching
                row['Merchant'],
                row['Category'],
                row['Subcategory'],
                parsed,  # Full parsed pattern with conditions
                tags  # List of tags
            ))
    return rules


def _expr_to_regex(match_expr: str) -> str:
    """Convert a .rules match expression to a regex pattern for legacy matching.

    Examples:
        contains("NETFLIX") -> NETFLIX
        regex("UBER(?!.*EATS)") -> UBER(?!.*EATS)
        contains("COSTCO") and amount > 200 -> COSTCO (amount ignored in regex)
    """
    import re as regex_module
    # Extract pattern from contains("...") or regex("...")
    contains_match = regex_module.search(r'contains\s*\(\s*["\']([^"\']+)["\']\s*\)', match_expr)
    if contains_match:
        return contains_match.group(1)

    regex_match = regex_module.search(r'regex\s*\(\s*["\']([^"\']+)["\']\s*\)', match_expr)
    if regex_match:
        return regex_match.group(1)

    # If no function found, try to extract a quoted string
    quoted_match = regex_module.search(r'["\']([^"\']+)["\']', match_expr)
    if quoted_match:
        return quoted_match.group(1)

    # Fallback: use the expression as-is (may not work)
    return match_expr


def get_all_rules(rules_path=None):
    """Get user-defined merchant rules.

    Args:
        rules_path: Optional path to user's merchants file (.rules or .csv)

    Returns:
        List of (pattern, merchant, category, subcategory, parsed_pattern, source, tags) tuples.
        Source is always 'user' for rules from the file.
    """
    user_rules_with_source = []
    if rules_path:
        # Check if it's the new .rules format
        if rules_path.endswith('.rules'):
            try:
                from .merchant_engine import load_merchants_file
                from pathlib import Path
                engine = load_merchants_file(Path(rules_path))
                # Convert MerchantRule objects to the tuple format used by parsing code
                for rule in engine.rules:
                    # Convert expression to regex for backward compatibility
                    regex_pattern = _expr_to_regex(rule.match_expr)
                    parsed = ParsedPattern(regex_pattern=regex_pattern)
                    user_rules_with_source.append((
                        regex_pattern,    # pattern (converted to regex for matching)
                        rule.name,        # merchant name
                        rule.category,
                        rule.subcategory,
                        parsed,
                        'user',
                        list(rule.tags)
                    ))
                return user_rules_with_source
            except Exception:
                pass  # Fall through to CSV handling if .rules parsing fails

        # CSV format (legacy)
        user_rules = load_merchant_rules(rules_path)
        # Add source='user' to each rule
        for rule in user_rules:
            if len(rule) == 6:
                # New format with tags
                pattern, merchant, category, subcategory, parsed, tags = rule
                user_rules_with_source.append((pattern, merchant, category, subcategory, parsed, 'user', tags))
            elif len(rule) == 5:
                # Old format without tags
                pattern, merchant, category, subcategory, parsed = rule
                user_rules_with_source.append((pattern, merchant, category, subcategory, parsed, 'user', []))
            else:
                pattern, merchant, category, subcategory = rule
                parsed = ParsedPattern(regex_pattern=pattern)
                user_rules_with_source.append((pattern, merchant, category, subcategory, parsed, 'user', []))

    return user_rules_with_source


def diagnose_rules(csv_path=None):
    """Get detailed diagnostic information about rule loading.

    Returns a dict with:
        - user_rules_path: Path to user rules file (or None)
        - user_rules_exists: Whether the user rules file exists
        - user_rules_count: Number of user rules loaded
        - user_rules: List of user rules (pattern, merchant, category, subcategory, tags)
        - user_rules_errors: List of any errors encountered while loading
        - total_rules: Total rules count (same as user_rules_count)
        - rules_with_tags: Count of rules that have tags
        - unique_tags: Set of all unique tags across all rules
    """
    import re

    result = {
        'user_rules_path': csv_path,
        'user_rules_exists': False,
        'user_rules_count': 0,
        'user_rules': [],
        'user_rules_errors': [],
        'total_rules': 0,
        'rules_with_tags': 0,
        'unique_tags': set(),
    }

    if not csv_path:
        return result

    result['user_rules_exists'] = os.path.exists(csv_path)

    if not result['user_rules_exists']:
        return result

    # Load user rules with detailed error tracking
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            result['file_size_bytes'] = len(raw_content)
            result['file_lines'] = raw_content.count('\n') + 1

        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            non_comment_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
            result['non_comment_lines'] = len(non_comment_lines)

            # Check for header
            if non_comment_lines:
                first_line = non_comment_lines[0].strip()
                if 'Pattern' in first_line and 'Merchant' in first_line:
                    result['has_header'] = True
                else:
                    result['has_header'] = False
                    result['user_rules_errors'].append(
                        f"Missing or invalid header. Expected 'Pattern,Merchant,Category,Subcategory', got: {first_line[:50]}"
                    )

        # Now load rules with validation
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Filter out comments AND empty lines
            lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
            reader = csv.DictReader(lines)

            row_num = 1  # Start after header
            for row in reader:
                row_num += 1
                pattern = row.get('Pattern', '').strip()

                if not pattern:
                    continue  # Skip empty patterns silently

                # Validate the regex pattern
                try:
                    re.compile(pattern, re.IGNORECASE)
                except re.error as e:
                    result['user_rules_errors'].append(
                        f"Row {row_num}: Invalid regex pattern '{pattern}': {e}"
                    )
                    continue

                merchant = row.get('Merchant', '').strip()
                category = row.get('Category', '').strip()
                subcategory = row.get('Subcategory', '').strip()

                # Parse tags (optional, pipe-separated)
                tags_str = row.get('Tags') or ''
                tags_str = tags_str.strip()
                tags = [t.strip() for t in tags_str.split('|') if t.strip()] if tags_str else []

                if not merchant:
                    result['user_rules_errors'].append(
                        f"Row {row_num}: Missing merchant name for pattern '{pattern}'"
                    )
                if not category:
                    result['user_rules_errors'].append(
                        f"Row {row_num}: Missing category for pattern '{pattern}'"
                    )

                result['user_rules'].append((pattern, merchant, category, subcategory, tags))

                # Track tag statistics
                if tags:
                    result['rules_with_tags'] += 1
                    result['unique_tags'].update(tags)

        result['user_rules_count'] = len(result['user_rules'])
        result['total_rules'] = result['user_rules_count']

    except Exception as e:
        result['user_rules_errors'].append(f"Failed to read file: {e}")

    return result


def clean_description(description, cleaning_patterns=None):
    """Clean and normalize raw transaction descriptions.

    Args:
        description: Raw transaction description
        cleaning_patterns: Optional list of regex patterns to strip from descriptions.
                          Loaded from settings.yaml 'description_cleaning' key.

    Returns:
        Cleaned description with patterns removed and whitespace normalized.
    """
    cleaned = description

    # Apply user-configured cleaning patterns
    if cleaning_patterns:
        for pattern in cleaning_patterns:
            try:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            except re.error:
                # Invalid regex, skip
                continue

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def extract_merchant_name(description, cleaning_patterns=None):
    """Extract a readable merchant name from a cleaned description.

    Used as fallback when no pattern matches.
    """
    cleaned = clean_description(description, cleaning_patterns)

    # Remove non-alphabetic characters for grouping, keep first 2-3 words
    words = re.sub(r'[^A-Za-z\s]', ' ', cleaned).split()[:3]

    if words:
        return ' '.join(words).title()
    return 'Unknown'


def normalize_merchant(
    description: str,
    rules: list,
    amount: Optional[float] = None,
    txn_date: Optional[date] = None,
    cleaning_patterns: Optional[List[str]] = None
) -> Tuple[str, str, str, Optional[dict]]:
    """Normalize a merchant description to (name, category, subcategory, match_info).

    Args:
        description: Raw transaction description
        rules: List of (pattern, merchant, category, subcategory, parsed_pattern, source, tags) tuples
              or older formats with fewer elements
        amount: Optional transaction amount for modifier matching
        txn_date: Optional transaction date for modifier matching
        cleaning_patterns: Optional list of regex patterns to strip from descriptions

    Returns:
        Tuple of (merchant_name, category, subcategory, match_info)
        match_info is a dict with 'pattern', 'source', 'tags', or None if no match
    """
    # Clean the description for better matching
    cleaned = clean_description(description, cleaning_patterns)
    desc_upper = description.upper()
    cleaned_upper = cleaned.upper()

    # Try pattern matching against both original and cleaned
    for rule in rules:
        # Handle various formats: 4-tuple, 5-tuple, 6-tuple, 7-tuple (with tags)
        tags = []
        if len(rule) == 7:
            pattern, merchant, category, subcategory, parsed, source, tags = rule
        elif len(rule) == 6:
            pattern, merchant, category, subcategory, parsed, source = rule
        elif len(rule) == 5:
            pattern, merchant, category, subcategory, parsed = rule
            source = 'unknown'
        else:
            pattern, merchant, category, subcategory = rule
            parsed = None
            source = 'unknown'

        try:
            # Check regex pattern first (most common case)
            regex_matches = (
                re.search(pattern, desc_upper, re.IGNORECASE) or
                re.search(pattern, cleaned_upper, re.IGNORECASE)
            )
            if not regex_matches:
                continue

            # If pattern has modifiers, check them
            if parsed and (parsed.amount_conditions or parsed.date_conditions):
                if not check_all_conditions(parsed, amount, txn_date):
                    continue

            return (merchant, category, subcategory, {'pattern': pattern, 'source': source, 'tags': tags})
        except re.error:
            # Invalid regex pattern, skip
            continue

    # Fallback: extract merchant name from description, categorize as Unknown
    merchant_name = extract_merchant_name(description, cleaning_patterns)
    return (merchant_name, 'Unknown', 'Unknown', None)


def explain_description(
    description: str,
    rules: list,
    amount: Optional[float] = None,
    txn_date: Optional[date] = None,
    cleaning_patterns: Optional[List[str]] = None
) -> dict:
    """Trace how a description is processed and matched.

    Returns a dict with detailed information about the matching process:
    - original: The original description
    - cleaned: The cleaned description (if different)
    - matched_rule: The rule that matched (if any)
    - merchant: Resulting merchant name
    - category: Resulting category
    - subcategory: Resulting subcategory
    - is_unknown: Whether this is an unknown merchant
    """
    # Use existing clean_description function
    cleaned = clean_description(description, cleaning_patterns)

    result = {
        'original': description,
        'cleaned': cleaned if cleaned != description else None,
        'matched_rule': None,
        'merchant': None,
        'category': None,
        'subcategory': None,
        'is_unknown': False,
    }

    # Try pattern matching against both original and cleaned
    desc_upper = description.upper()
    cleaned_upper = cleaned.upper()

    for rule in rules:
        # Handle various formats
        tags = []
        if len(rule) == 7:
            pattern, merchant, category, subcategory, parsed, source, tags = rule
        elif len(rule) == 6:
            pattern, merchant, category, subcategory, parsed, source = rule
        elif len(rule) == 5:
            pattern, merchant, category, subcategory, parsed = rule
            source = 'unknown'
        else:
            pattern, merchant, category, subcategory = rule
            parsed = None
            source = 'unknown'

        try:
            # Check regex pattern
            match_on_original = re.search(pattern, desc_upper, re.IGNORECASE)
            match_on_cleaned = re.search(pattern, cleaned_upper, re.IGNORECASE)

            if not (match_on_original or match_on_cleaned):
                continue

            # If pattern has modifiers, check them
            if parsed and (parsed.amount_conditions or parsed.date_conditions):
                if not check_all_conditions(parsed, amount, txn_date):
                    continue

            result['matched_rule'] = {
                'pattern': pattern,
                'source': source,
                'matched_on': 'original' if match_on_original else 'cleaned',
                'tags': tags,
            }
            result['merchant'] = merchant
            result['category'] = category
            result['subcategory'] = subcategory
            return result

        except re.error:
            continue

    # No match - unknown merchant
    result['is_unknown'] = True
    result['merchant'] = extract_merchant_name(description, cleaning_patterns)
    result['category'] = 'Unknown'
    result['subcategory'] = 'Unknown'
    return result
