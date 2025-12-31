"""
Classification rule parser and evaluator.

Parses rules like: category=Bills[months>=6] -> monthly,avg
Evaluates them against merchant statistics to determine bucket and calc_type.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


# Default rules that replicate current hardcoded behavior
DEFAULT_RULES = """# Tally Classification Rules
# Syntax: CONDITION[modifier] -> bucket,calc_type
#
# Conditions:
#   category=X           Match category
#   subcategory=X        Match subcategory
#   category=X,subcategory=Y  Match both
#
# Modifiers (in square brackets):
#   [months>=N]          Month count threshold
#   [months>=N%]         Percentage of data period (50% of 12 months = 6)
#   [count<=N]           Transaction count
#   [total>N]            Total amount
#   [cv>N]               Coefficient of variation (payment consistency)
#   [max>N]              Maximum single payment
#   [max_avg_ratio>N]    Ratio of max to average payment
#
# Buckets: travel, annual, periodic, monthly, one_off, variable
# Calc types: avg (when active), /12 (annual spread), auto (cv-based)
#
# Note: Transactions are excluded from spending via special tags (income, refund, transfer)
# rather than categories. See 'tally workflow' for details.

# Travel
category=Travel -> travel,/12

# Annual bills (once or twice a year)
category=Bills,subcategory=Insurance[months<=2][count<=2] -> annual,/12
category=Bills,subcategory=Tax[months<=2] -> annual,/12
category=Bills,subcategory=Membership[months<=2] -> annual,/12
category=Family,subcategory=Charity -> annual,/12
category=Charity,subcategory=Donation -> annual,/12

# Periodic (quarterly, treatment series, lumpy)
category=Education,subcategory=Tuition -> periodic,/12
category=Bills,subcategory=Insurance[months>=3] -> periodic,/12
category=Health,subcategory=Medical[months>=2] -> periodic,/12
category=Health,subcategory=Dental[months>=2] -> periodic,/12
category=Health,subcategory=Orthodontics[months>=2] -> periodic,/12
category=Bills[total>5000][cv>0.8][max_avg_ratio>3] -> periodic,/12
category=Education[total>5000][cv>0.8][max_avg_ratio>3] -> periodic,/12
category=Health[total>5000][cv>0.8][max_avg_ratio>3] -> periodic,/12

# Monthly recurring (bills: 50% threshold, services: 75%)
category=Bills[months>=50%] -> monthly,auto
category=Utilities[months>=50%] -> monthly,auto
category=Subscriptions[months>=50%] -> monthly,auto
category=Home,subcategory=Lawn[months>=75%] -> monthly,auto
category=Home,subcategory=Security[months>=75%] -> monthly,auto
category=Home,subcategory=Cleaning[months>=75%] -> monthly,auto
category=Health,subcategory=Gym[months>=75%] -> monthly,auto
category=Health,subcategory=Fitness[months>=75%] -> monthly,auto
category=Health,subcategory=Pharmacy[months>=75%] -> monthly,auto
category=Food,subcategory=Grocery[months>=75%] -> monthly,auto
category=Food,subcategory=Delivery[months>=75%] -> monthly,auto
category=Transport,subcategory=Gas[months>=75%] -> monthly,auto
category=Transport,subcategory=Parking[months>=75%] -> monthly,auto
category=Transport,subcategory=Transit[months>=75%] -> monthly,auto
category=Personal,subcategory=Childcare[months>=75%] -> monthly,auto
category=Personal,subcategory=Services[months>=75%] -> monthly,auto
category=Personal,subcategory=Grooming[months>=75%] -> monthly,auto

# One-off (high value, infrequent)
subcategory=Procedure -> one_off,/12
category=Shopping[months<=3][total>1000] -> one_off,/12
category=Home[months<=3][total>1000] -> one_off,/12
category=Personal[months<=3][total>1000] -> one_off,/12
subcategory=Improvement[months<=3][total>1000] -> one_off,/12
subcategory=Appliance[months<=3][total>1000] -> one_off,/12
subcategory=HVAC[months<=3][total>1000] -> one_off,/12
subcategory=Repair[months<=3][total>1000] -> one_off,/12
subcategory=Furniture[months<=3][total>1000] -> one_off,/12
subcategory=Electronics[months<=3][total>1000] -> one_off,/12
subcategory=Jewelry[months<=3][total>1000] -> one_off,/12
subcategory=Luxury[months<=3][total>1000] -> one_off,/12

# Variable: frequent (50%+ months) and consistent (CV < 0.3) -> use average
[months>=50%][cv<0.3] -> variable,avg

# Default: everything else is variable with /12
* -> variable,/12
"""


@dataclass
class NumericCondition:
    """A numeric condition like [months>=6] or [cv>0.8]."""
    variable: str          # 'months', 'count', 'total', 'cv', 'max', 'avg', 'max_avg_ratio'
    operator: str          # '>', '<', '>=', '<=', '='
    value: float
    is_percentage: bool = False  # True if value should be multiplied by num_months


@dataclass
class FieldMatch:
    """A field equality match like category=Bills."""
    field: str             # 'category' or 'subcategory'
    value: str             # The value to match


@dataclass
class ClassificationRule:
    """A complete classification rule."""
    line_number: int
    raw_text: str
    field_matches: List[FieldMatch]
    conditions: List[NumericCondition]
    bucket: str            # 'travel', 'annual', 'periodic', 'monthly', 'one_off', 'variable'
    calc_type: str         # 'avg', '/12', 'auto'
    is_default: bool = False  # True for wildcard rule (*)


class RuleParseError(ValueError):
    """Error parsing a classification rule."""
    pass


# Regex patterns for parsing
RULE_PATTERN = re.compile(r'^(.+?)\s*->\s*(\w+)\s*,\s*(\w+|/12)\s*$')
FIELD_MATCH = re.compile(r'(\w+)=([^,\[\]]+)')
MODIFIER = re.compile(r'\[(\w+)(>=|<=|>|<|=)(\d+\.?\d*)(%?)\]')

# Valid values
VALID_BUCKETS = {'travel', 'annual', 'periodic', 'monthly', 'one_off', 'variable'}
VALID_CALC_TYPES = {'avg', '/12', 'auto'}
VALID_VARIABLES = {'months', 'count', 'total', 'cv', 'max', 'avg', 'max_avg_ratio'}


def parse_rule(line: str, line_number: int) -> Optional[ClassificationRule]:
    """Parse a single rule line into a ClassificationRule object."""
    # Skip empty lines and comments
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None

    # Parse the basic structure: conditions -> bucket,calc_type
    match = RULE_PATTERN.match(stripped)
    if not match:
        raise RuleParseError(f"Line {line_number}: Invalid rule syntax: {stripped}")

    condition_part, bucket, calc_type = match.groups()

    # Validate bucket
    if bucket not in VALID_BUCKETS:
        raise RuleParseError(f"Line {line_number}: Invalid bucket '{bucket}'. Must be one of: {VALID_BUCKETS}")

    # Validate calc_type
    if calc_type not in VALID_CALC_TYPES:
        raise RuleParseError(f"Line {line_number}: Invalid calc_type '{calc_type}'. Must be one of: {VALID_CALC_TYPES}")

    # Check for wildcard rule
    if condition_part.strip() == '*':
        return ClassificationRule(
            line_number=line_number,
            raw_text=stripped,
            field_matches=[],
            conditions=[],
            bucket=bucket,
            calc_type=calc_type,
            is_default=True
        )

    # Extract modifiers (everything in [...])
    conditions = []
    for mod_match in MODIFIER.finditer(condition_part):
        var, op, value, is_pct = mod_match.groups()
        if var not in VALID_VARIABLES:
            raise RuleParseError(f"Line {line_number}: Invalid variable '{var}'. Must be one of: {VALID_VARIABLES}")
        conditions.append(NumericCondition(
            variable=var,
            operator=op,
            value=float(value),
            is_percentage=bool(is_pct)
        ))

    # Remove modifiers from condition_part to parse field matches
    field_part = MODIFIER.sub('', condition_part)

    # Parse field matches
    field_matches = []
    for field_match in FIELD_MATCH.finditer(field_part):
        field_name, field_value = field_match.groups()
        if field_name not in ('category', 'subcategory'):
            raise RuleParseError(f"Line {line_number}: Invalid field '{field_name}'. Must be 'category' or 'subcategory'")
        field_matches.append(FieldMatch(field=field_name, value=field_value))

    return ClassificationRule(
        line_number=line_number,
        raw_text=stripped,
        field_matches=field_matches,
        conditions=conditions,
        bucket=bucket,
        calc_type=calc_type,
        is_default=False
    )


def parse_rules(text: str) -> List[ClassificationRule]:
    """Parse rules from a string."""
    rules = []
    for line_num, line in enumerate(text.strip().split('\n'), start=1):
        try:
            rule = parse_rule(line, line_num)
            if rule:
                rules.append(rule)
        except RuleParseError:
            raise  # Re-raise with line context
    return rules


def load_rules(filepath: str) -> List[ClassificationRule]:
    """Load classification rules from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return parse_rules(f.read())


def get_default_rules() -> str:
    """Return the built-in default rules as a string."""
    return DEFAULT_RULES


def get_default_rules_parsed() -> List[ClassificationRule]:
    """Return the built-in default rules as parsed objects."""
    return parse_rules(DEFAULT_RULES)


def evaluate_condition(cond: NumericCondition, stats: Dict[str, Any], num_months: int) -> bool:
    """Evaluate a single numeric condition against merchant stats."""
    # Get the variable value
    if cond.variable == 'months':
        value = stats.get('months_active', 1)
    elif cond.variable == 'count':
        value = stats.get('count', 0)
    elif cond.variable == 'total':
        value = stats.get('total', 0)
    elif cond.variable == 'cv':
        value = stats.get('cv', 0)
    elif cond.variable == 'max':
        value = stats.get('max_payment', 0)
    elif cond.variable == 'avg':
        count = stats.get('count', 1)
        value = stats.get('total', 0) / count if count > 0 else 0
    elif cond.variable == 'max_avg_ratio':
        count = stats.get('count', 1)
        avg = stats.get('total', 0) / count if count > 0 else 0
        max_payment = stats.get('max_payment', 0)
        value = max_payment / avg if avg > 0 else 0
    else:
        return False

    # Get threshold (handle percentage)
    threshold = cond.value
    if cond.is_percentage:
        # Convert percentage to absolute based on num_months
        # e.g., 50% with 12 months = 6, minimum 2
        threshold = max(2, int(num_months * (cond.value / 100.0)))

    # Compare
    if cond.operator == '>':
        return value > threshold
    elif cond.operator == '>=':
        return value >= threshold
    elif cond.operator == '<':
        return value < threshold
    elif cond.operator == '<=':
        return value <= threshold
    elif cond.operator == '=':
        return abs(value - threshold) < 0.001
    return False


def matches_rule(rule: ClassificationRule, stats: Dict[str, Any], num_months: int) -> bool:
    """Check if a rule matches the given merchant stats."""
    # Default rule matches everything
    if rule.is_default:
        return True

    # Check field matches (all must match)
    for fm in rule.field_matches:
        if fm.field == 'category':
            if stats.get('category', '') != fm.value:
                return False
        elif fm.field == 'subcategory':
            if stats.get('subcategory', '') != fm.value:
                return False

    # Check numeric conditions (all must match)
    for cond in rule.conditions:
        if not evaluate_condition(cond, stats, num_months):
            return False

    return True


def resolve_calc_type(calc_type: str, cv: float) -> str:
    """Resolve 'auto' calc_type based on CV."""
    if calc_type == 'auto':
        # auto: use avg if consistent (CV < 0.3), /12 otherwise
        return 'avg' if cv < 0.3 else '/12'
    return calc_type


def classify_merchant(
    stats: Dict[str, Any],
    rules: List[ClassificationRule],
    num_months: int = 12
) -> Tuple[str, str]:
    """
    Classify a merchant using the rule engine.

    Args:
        stats: Dict with keys: category, subcategory, months_active, count, total, cv, max_payment
        rules: List of ClassificationRule objects
        num_months: Number of months in the data period

    Returns: (bucket, calc_type)
    """
    for rule in rules:
        if matches_rule(rule, stats, num_months):
            # Resolve calc_type
            cv = stats.get('cv', 0)
            calc_type = resolve_calc_type(rule.calc_type, cv)
            return (rule.bucket, calc_type)

    # Should never reach here if rules include a default
    return ('variable', '/12')


def write_default_rules(filepath: str) -> None:
    """Write the default rules to a file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_RULES, encoding='utf-8')
