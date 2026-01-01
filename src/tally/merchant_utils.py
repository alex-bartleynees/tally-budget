"""
Merchant normalization utilities for spending analysis.

This module provides functions to clean and categorize merchant descriptions
from credit card and bank statements.
"""

import csv
import os
import re
from datetime import date
from typing import Optional, List, Tuple, Dict

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

    For new functions (normalized, anyof, startswith, fuzzy), we return the
    full expression since these need to be evaluated by the expression parser.

    Examples:
        contains("NETFLIX") -> NETFLIX
        regex("UBER(?!.*EATS)") -> UBER(?!.*EATS)
        contains("COSTCO") and amount > 200 -> COSTCO (amount ignored in regex)
        normalized("UBEREATS") -> normalized("UBEREATS")  # preserved for expr parser
    """
    import re as regex_module

    # Check if expression uses new functions that need to be preserved
    if regex_module.search(r'\b(normalized|anyof|startswith|fuzzy)\s*\(', match_expr):
        # Return full expression - will be handled by expression parser
        return match_expr

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
                # Only include categorization rules (skip tag-only rules)
                for rule in engine.rules:  # Include ALL rules (categorization + tag-only)
                    # Preserve the full match_expr for expression-based rules
                    # This allows amount/date conditions like "regex(...) and amount == 1500" to work
                    pattern = rule.match_expr
                    regex_pattern = _expr_to_regex(rule.match_expr)
                    parsed = ParsedPattern(regex_pattern=regex_pattern)
                    user_rules_with_source.append((
                        pattern,          # Full expression (for expr matching)
                        rule.name,        # merchant name
                        rule.category,    # Empty for tag-only rules
                        rule.subcategory, # Empty for tag-only rules
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


def get_tag_only_rules(rules_path):
    """Get tag-only rules from a .rules file.

    Tag-only rules add tags to transactions without changing their category.

    Returns:
        List of MerchantRule objects that are tag-only (no category).
    """
    if not rules_path or not rules_path.endswith('.rules'):
        return []

    try:
        from .merchant_engine import load_merchants_file
        from pathlib import Path
        engine = load_merchants_file(Path(rules_path))
        return engine.tag_only_rules
    except Exception:
        return []


def apply_tag_rules(transaction, tag_rules):
    """Apply tag-only rules to a transaction, adding matching tags.

    Args:
        transaction: Transaction dict with description, amount, date, field, source
        tag_rules: List of MerchantRule objects (tag-only rules)

    Returns:
        List of additional tags from matching tag-only rules.
    """
    from tally import expr_parser

    additional_tags = []
    description = transaction.get('raw_description', transaction.get('description', ''))
    amount = transaction.get('amount', 0)
    txn_date = transaction.get('date')
    if txn_date and hasattr(txn_date, 'date'):
        txn_date = txn_date.date()
    field = transaction.get('field')
    source = transaction.get('source')

    for rule in tag_rules:
        try:
            # Build context for expression matching
            ctx = {
                'description': description,
                'amount': amount,
                'field': field,
                'source': source,
            }
            if txn_date:
                ctx['date'] = txn_date

            if expr_parser.matches_transaction(rule.match_expr, ctx):
                # Resolve dynamic tags
                resolved = _resolve_dynamic_tags(list(rule.tags), ctx)
                additional_tags.extend(resolved)
        except Exception:
            continue

    return additional_tags


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
    cleaning_patterns: Optional[List[str]] = None,
    field: Optional[Dict[str, str]] = None,
    data_source: Optional[str] = None,
) -> Tuple[str, str, str, Optional[dict]]:
    """Normalize a merchant description to (name, category, subcategory, match_info).

    Two-pass matching:
    1. First categorization rule that matches sets merchant/category/subcategory
    2. Tags are collected from ALL matching rules (including tag-only rules)

    Args:
        description: Raw transaction description
        rules: List of (pattern, merchant, category, subcategory, parsed_pattern, source, tags) tuples
              or older formats with fewer elements
        amount: Optional transaction amount for modifier matching
        txn_date: Optional transaction date for modifier matching
        cleaning_patterns: Optional list of regex patterns to strip from descriptions
        field: Optional dict of custom CSV captures (for field.name in rule expressions)
        data_source: Optional data source name (for source variable in rule expressions and dynamic tags)

    Returns:
        Tuple of (merchant_name, category, subcategory, match_info)
        match_info is a dict with 'pattern', 'source', 'tags', or None if no match
    """
    from tally import expr_parser

    # Clean the description for better matching
    cleaned = clean_description(description, cleaning_patterns)
    desc_upper = description.upper()
    cleaned_upper = cleaned.upper()

    # Build transaction context once
    transaction = {'description': description, 'amount': amount or 0, 'field': field, 'source': data_source}
    if txn_date:
        transaction['date'] = txn_date
    transaction_cleaned = {'description': cleaned, 'amount': amount or 0, 'field': field, 'source': data_source}
    if txn_date:
        transaction_cleaned['date'] = txn_date

    # Result from first categorization match
    result_merchant = None
    result_category = None
    result_subcategory = None
    result_pattern = None
    result_source = None

    # Collect tags from ALL matching rules
    all_tags = []

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
            # Check if rule matches
            matches = False

            if _is_expression_pattern(pattern):
                # Use expression parser for expression-based rules
                matches = (expr_parser.matches_transaction(pattern, transaction) or
                          expr_parser.matches_transaction(pattern, transaction_cleaned))
            else:
                # Legacy regex pattern matching
                regex_matches = (
                    re.search(pattern, desc_upper, re.IGNORECASE) or
                    re.search(pattern, cleaned_upper, re.IGNORECASE)
                )
                if regex_matches:
                    # Check modifiers if present
                    if parsed and (parsed.amount_conditions or parsed.date_conditions):
                        matches = check_all_conditions(parsed, amount, txn_date)
                    else:
                        matches = True

            if not matches:
                continue

            # Rule matched - collect tags
            if tags:
                resolved_tags = _resolve_dynamic_tags(tags, transaction)
                all_tags.extend(resolved_tags)

            # Use first categorization rule for merchant/category/subcategory
            # Tag-only rules have empty category
            if result_merchant is None and category:
                result_merchant = merchant
                result_category = category
                result_subcategory = subcategory
                result_pattern = pattern
                result_source = source

        except (re.error, expr_parser.ExpressionError):
            # Invalid pattern, skip
            continue

    # Return matched result with all collected tags (deduplicated, order preserved)
    unique_tags = list(dict.fromkeys(all_tags))
    if result_merchant is not None:
        return (result_merchant, result_category, result_subcategory,
                {'pattern': result_pattern, 'source': result_source, 'tags': unique_tags})

    # Fallback: extract merchant name from description, categorize as Unknown
    # Still include any tags from tag-only rules that matched
    merchant_name = extract_merchant_name(description, cleaning_patterns)
    if unique_tags:
        return (merchant_name, 'Unknown', 'Unknown', {'pattern': None, 'source': 'auto', 'tags': unique_tags})
    return (merchant_name, 'Unknown', 'Unknown', None)


def _is_expression_pattern(pattern: str) -> bool:
    """Check if a pattern is an expression (uses function syntax) vs a regex."""
    import re
    # Expression patterns start with:
    # - Function calls like contains(), normalized(), extract(), etc.
    # - Field access like field.txn_type
    # - Boolean operators like 'and', 'or'
    # - Parenthesized expressions
    # - Variable comparisons like amount > 500, month == 12, source == "Amex"
    function_pattern = r'^(contains|normalized|anyof|startswith|fuzzy|regex|extract|split|substring|trim|exists)\s*\('
    variable_pattern = r'^(amount|month|year|day|source|description)\s*[<>=!]'
    return bool(re.match(function_pattern, pattern)) or \
           bool(re.match(variable_pattern, pattern)) or \
           pattern.startswith('field.') or \
           ' and ' in pattern or ' or ' in pattern or pattern.startswith('(')


def _resolve_dynamic_tags(
    tags: List[str],
    transaction: Dict,
) -> List[str]:
    """Resolve dynamic tags, evaluating any {expression} placeholders.

    Supports dynamic tag values from field access or extraction functions:
        ["wire", "banking"]           -> ["wire", "banking"]
        ["{field.txn_type}"]          -> ["ach"] (if field.txn_type == "ACH")
        ["banking", "{field.type}"]   -> ["banking", "wire"]
        ["{extract(field.memo, r'PROJ:(\\w+)')}"] -> ["abc123"]

    Empty values from expressions are skipped.
    All tags are lowercased for consistency.

    Args:
        tags: List of tag strings, may contain {expression} placeholders
        transaction: Transaction dict with description, amount, date, field

    Returns:
        List of resolved tag strings (lowercased)
    """
    from tally import expr_parser

    resolved = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue

        if tag.startswith('{') and tag.endswith('}'):
            # Dynamic tag - evaluate expression
            expr = tag[1:-1].strip()  # Remove { }
            if not expr:
                continue

            try:
                ctx = expr_parser.TransactionContext.from_transaction(transaction)
                tree = expr_parser.parse_expression(expr)
                evaluator = expr_parser.TransactionEvaluator(ctx)
                value = evaluator.evaluate(tree)
                if value:  # Only add non-empty values
                    stripped = str(value).strip()
                    if stripped:  # Skip whitespace-only values
                        resolved.append(stripped.lower())
            except expr_parser.ExpressionError:
                # Skip invalid expressions silently
                pass
        else:
            # Static tag
            resolved.append(tag.lower())

    return resolved


def explain_description(
    description: str,
    rules: list,
    amount: Optional[float] = None,
    txn_date: Optional[date] = None,
    cleaning_patterns: Optional[List[str]] = None,
    field: Optional[Dict[str, str]] = None,
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
    from tally import expr_parser

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
            # Determine if this is an expression pattern or a regex pattern
            if _is_expression_pattern(pattern):
                # Use expression parser for expression-based rules
                transaction = {'description': description, 'amount': amount or 0, 'field': field}
                if txn_date:
                    transaction['date'] = txn_date
                match_on_original = expr_parser.matches_transaction(pattern, transaction)

                # Also try with cleaned description
                transaction_cleaned = {'description': cleaned, 'amount': amount or 0, 'field': field}
                if txn_date:
                    transaction_cleaned['date'] = txn_date
                match_on_cleaned = expr_parser.matches_transaction(pattern, transaction_cleaned)

                if not (match_on_original or match_on_cleaned):
                    continue
            else:
                # Legacy regex pattern matching
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

        except (re.error, expr_parser.ExpressionError):
            continue

    # No match - unknown merchant
    result['is_unknown'] = True
    result['merchant'] = extract_merchant_name(description, cleaning_patterns)
    result['category'] = 'Unknown'
    result['subcategory'] = 'Unknown'
    return result
