"""
Tally CLI - Command-line interface.

Usage:
    tally /path/to/config/dir               # Analyze using config directory
    tally /path/to/config/dir --summary     # Summary only (no HTML)
    tally /path/to/config/dir --settings settings-2024.yaml
    tally --help-config                     # Show detailed config documentation
"""

import argparse
import os
import shutil
import sys

# Terminal color support
def _supports_color():
    """Check if the terminal supports color output."""
    if not sys.stdout.isatty():
        return False
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    # Check for common terminal types
    term = os.environ.get('TERM', '')
    return term != 'dumb'

def _setup_windows_encoding():
    """Set UTF-8 encoding on Windows to support Unicode output."""
    if sys.platform != 'win32':
        return

    import codecs

    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        # Skip if already UTF-8
        if getattr(stream, 'encoding', '').lower().replace('-', '') == 'utf8':
            continue
        try:
            # Method 1: reconfigure (works in normal Python 3.7+)
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError):
            try:
                # Method 2: Use codecs writer (more reliable for PyInstaller)
                if hasattr(stream, 'buffer'):
                    writer = codecs.getwriter('utf-8')(stream.buffer, errors='replace')
                    writer.encoding = 'utf-8'
                    setattr(sys, stream_name, writer)
            except Exception:
                pass

_setup_windows_encoding()

class _Colors:
    """ANSI color codes with automatic detection."""
    def __init__(self):
        if _supports_color():
            self.RESET = '\033[0m'
            self.BOLD = '\033[1m'
            self.DIM = '\033[2m'
            self.GREEN = '\033[32m'
            self.CYAN = '\033[36m'
            self.BLUE = '\033[34m'
            self.YELLOW = '\033[33m'
            self.UNDERLINE = '\033[4m'
        else:
            self.RESET = ''
            self.BOLD = ''
            self.DIM = ''
            self.GREEN = ''
            self.CYAN = ''
            self.BLUE = ''
            self.YELLOW = ''
            self.UNDERLINE = ''

C = _Colors()

from ._version import (
    VERSION, GIT_SHA, REPO_URL, check_for_updates,
    get_latest_release_info, perform_update
)
from .config_loader import load_config

BANNER = ''
from .merchant_utils import get_all_rules, diagnose_rules, explain_description
from .analyzer import (
    parse_amex,
    parse_boa,
    parse_generic_csv,
    auto_detect_csv_format,
    analyze_transactions,
    print_summary,
    write_summary_file,
    write_summary_file_vue,
)


CONFIG_HELP = '''
BUDGET ANALYZER - CONFIGURATION
================================

QUICK START
-----------
1. Run: tally init ./my-budget
2. Add CSV/TXT statements to my-budget/data/
3. Edit my-budget/config/settings.yaml with your data sources
4. Run: tally run ./my-budget/config

DIRECTORY STRUCTURE
-------------------
my-budget/
├── config/
│   ├── settings.yaml           # Data sources & settings
│   └── merchant_categories.csv # Custom category overrides (optional)
├── data/                       # Your statement exports
└── output/                     # Generated reports

SETTINGS.YAML
-------------
year: 2025
data_sources:
  - name: AMEX
    file: data/amex.csv
    type: amex                  # or "boa" for Bank of America
  # Custom CSV format:
  - name: Chase
    file: data/chase.csv
    format: "{date:%m/%d/%Y},{description},{amount}"
output_dir: output

# Optional: specify home locations (auto-detected if not set)
home_locations:
  - WA
  - OR                          # Nearby state to not count as travel

# Optional: pretty names for travel destinations
travel_labels:
  HI: Hawaii
  GB: United Kingdom

TRAVEL DETECTION
----------------
International transactions are automatically classified as travel.
Domestic out-of-state is NOT auto-travel. To opt-in, add merchant rules:

  .*\\sHI$,Hawaii Trip,Travel,Hawaii
  .*\\sCA$,California Trip,Travel,California

DISCOVERING UNKNOWN MERCHANTS
-----------------------------
Use the discover command to find uncategorized transactions:
  tally discover               # Human-readable output
  tally discover --format csv  # CSV output to copy-paste
  tally discover --format json # JSON for programmatic use

MERCHANT RULES
--------------
Define merchant patterns in merchant_categories.csv:

Pattern,Merchant,Category,Subcategory
MY LOCAL CAFE,Local Cafe,Food,Coffee
ZELLE.*JANE,Jane (Babysitter),Personal,Childcare

Pattern syntax (Python regex):
  NETFLIX           Contains "NETFLIX"
  DELTA|SOUTHWEST   Either one
  COSTCO(?!.*GAS)   COSTCO but not COSTCO GAS
  ^ATT\\s           Starts with "ATT "

Use: tally inspect <file.csv> to see transaction formats.
'''

STARTER_SETTINGS = '''# Tally Settings
year: {year}
title: "Spending Analysis {year}"

# Data sources - add your statement files here
data_sources:
  # Example AMEX CSV export:
  # - name: AMEX
  #   file: data/amex-{year}.csv
  #   type: amex
  #
  # Example Bank of America text statement:
  # - name: BOA Checking
  #   file: data/boa-checking.txt
  #   type: boa

output_dir: output
html_filename: spending_summary.html

# Home locations (auto-detected if not specified)
# Transactions outside these locations are classified as travel
# home_locations:
#   - WA
#   - OR

# Optional: pretty names for travel destinations in reports
# travel_labels:
#   HI: Hawaii
#   GB: United Kingdom

# Description cleaning patterns (regex)
# Strip payment processor prefixes/suffixes before matching merchant rules
# This simplifies your merchant patterns - no need to handle every variation
# description_cleaning:
#   - "^APLPAY\\\\s+"           # Apple Pay prefix
#   - "^SQ\\\\s*\\\\*"          # Square prefix
#   - "^TST\\\\*\\\\s*"         # Toast POS prefix
#   - "^PP\\\\s*\\\\*"          # PayPal prefix
#   - "\\\\s+DES:.*$"           # Bank of America DES suffix
#   - "\\\\s+ID:.*$"            # Bank of America ID suffix
'''

STARTER_MERCHANT_CATEGORIES = '''# Merchant Categorization Rules
#
# Define your merchant categorization rules here.
# Format: Pattern,Merchant,Category,Subcategory
#
# - Pattern: Python regex (case-insensitive) matched against transaction descriptions
# - Use | for alternatives: DELTA|SOUTHWEST matches either
# - Use (?!...) for negative lookahead: UBER\\s(?!EATS) excludes Uber Eats
# - Test patterns at regex101.com (Python flavor)
#
# First match wins.
# Run: tally inspect <file> to see your transaction descriptions.
#
# Examples:
#   MY LOCAL BAKERY,My Favorite Bakery,Food,Restaurant
#   JOHNS PLUMBING,John's Plumbing,Bills,Home Repair
#   ZELLE.*JANE,Jane (Babysitter),Personal,Childcare

Pattern,Merchant,Category,Subcategory

# Add your custom rules below:

'''

def find_config_dir():
    """Find the config directory, checking environment and both layouts.

    Resolution order:
    1. TALLY_CONFIG environment variable (if set and exists)
    2. ./config (old layout - config in current directory)
    3. ./tally/config (new layout - config in tally subdirectory)

    Note: Migration prompts are handled separately by run_migrations()
    during 'tally update', not here.

    Returns None if no config directory is found.
    """
    # Check environment variable first
    env_config = os.environ.get('TALLY_CONFIG')
    if env_config:
        env_path = os.path.abspath(env_config)
        if os.path.isdir(env_path):
            return env_path

    # Check old layout (backwards compatibility)
    # Note: Migration prompts are handled by run_migrations() during 'tally update'
    old_layout = os.path.abspath('config')
    if os.path.isdir(old_layout):
        return old_layout

    # Check new layout
    new_layout = os.path.abspath(os.path.join('tally', 'config'))
    if os.path.isdir(new_layout):
        return new_layout

    return None


# Schema version for asset migrations
SCHEMA_VERSION = 1


def get_schema_version(config_dir):
    """Get current schema version from config directory.

    Returns:
        int: Schema version (0 if no marker file exists - legacy layout)
    """
    schema_file = os.path.join(config_dir, '.tally-schema')
    if os.path.exists(schema_file):
        try:
            with open(schema_file, encoding='utf-8') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return 0
    return 0


def run_migrations(config_dir, skip_confirm=False):
    """Run any pending migrations on the config directory.

    Args:
        config_dir: Path to current config directory
        skip_confirm: If True, skip confirmation prompts (--yes flag)

    Returns:
        str: Path to config directory (may change if layout migrated)
    """
    current = get_schema_version(config_dir)

    if current >= SCHEMA_VERSION:
        return config_dir  # Already up to date

    # Run migrations in order
    if current < 1:
        result = migrate_v0_to_v1(config_dir, skip_confirm)
        if result:
            config_dir = result

    return config_dir


def migrate_v0_to_v1(old_config_dir, skip_confirm=False):
    """Migrate from legacy layout (./config) to new layout (./tally/config).

    Args:
        old_config_dir: Path to the old config directory
        skip_confirm: If True, skip confirmation prompt

    Returns:
        str: Path to new config directory, or None if user declined
    """
    # Only migrate if we're in the old layout (./config at working directory root)
    if os.path.basename(old_config_dir) != 'config':
        return None
    if os.path.dirname(old_config_dir) != os.getcwd():
        return None

    # Prompt user (skip if non-interactive or --yes flag)
    if not skip_confirm:
        # In non-interactive mode (e.g., LLM/CI), skip migration silently
        if not sys.stdin.isatty():
            return None

        print()
        print("Migration available: Layout update")
        print("  Current: ./config (legacy layout)")
        print("  New: ./tally/config")
        print()
        try:
            response = input("Migrate to new layout? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipped.")
            return None
        if response == 'n':
            return None

    # Perform migration
    tally_dir = os.path.abspath('tally')
    try:
        os.makedirs(tally_dir, exist_ok=True)

        # Move config directory
        new_config = os.path.join(tally_dir, 'config')
        print(f"  Moving config/ -> tally/config/")
        shutil.move(old_config_dir, new_config)

        # Move data and output directories if they exist
        for subdir in ['data', 'output']:
            old_path = os.path.abspath(subdir)
            if os.path.isdir(old_path):
                new_path = os.path.join(tally_dir, subdir)
                print(f"  Moving {subdir}/ -> tally/{subdir}/")
                shutil.move(old_path, new_path)

        # Write schema version marker
        schema_file = os.path.join(new_config, '.tally-schema')
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write('1\n')

        print("✓ Migrated to ./tally/")
        return new_config

    except (OSError, shutil.Error) as e:
        print(f"Error during migration: {e}", file=sys.stderr)
        return None


def init_config(target_dir):
    """Initialize a new config directory with starter files."""
    import datetime

    config_dir = os.path.join(target_dir, 'config')
    data_dir = os.path.join(target_dir, 'data')
    output_dir = os.path.join(target_dir, 'output')

    # Create directories
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    current_year = datetime.datetime.now().year
    files_created = []
    files_skipped = []

    # Write settings.yaml
    settings_path = os.path.join(config_dir, 'settings.yaml')
    if not os.path.exists(settings_path):
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(STARTER_SETTINGS.format(year=current_year))
        files_created.append('config/settings.yaml')
    else:
        files_skipped.append('config/settings.yaml')

    # Write merchant_categories.csv
    merchants_path = os.path.join(config_dir, 'merchant_categories.csv')
    if not os.path.exists(merchants_path):
        with open(merchants_path, 'w', encoding='utf-8') as f:
            f.write(STARTER_MERCHANT_CATEGORIES)
        files_created.append('config/merchant_categories.csv')
    else:
        files_skipped.append('config/merchant_categories.csv')

    # Create .gitignore for data privacy
    gitignore_path = os.path.join(target_dir, '.gitignore')
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write('''# Tally - Ignore sensitive data
data/
output/
''')
        files_created.append('.gitignore')

    return files_created, files_skipped


def cmd_init(args):
    """Handle the 'init' subcommand."""
    target_dir = os.path.abspath(args.dir)
    # Use relative paths for display
    rel_target = os.path.relpath(target_dir)
    if rel_target == '.':
        rel_target = './'

    print(f"Initializing budget directory: {C.BOLD}{rel_target}{C.RESET}")
    print()

    created, skipped = init_config(target_dir)

    # Show each file with its status
    all_files = [(f, True) for f in created] + [(f, False) for f in skipped]
    # Sort by filename for consistent ordering
    all_files.sort(key=lambda x: x[0])

    for f, was_created in all_files:
        if was_created:
            print(f"  {C.GREEN}✓{C.RESET} {f}")
        else:
            print(f"  {C.YELLOW}→{C.RESET} {C.DIM}{f} (exists){C.RESET}")

    # Check if data sources are configured in settings.yaml
    import yaml
    has_data_sources = False
    settings_path = os.path.join(target_dir, 'config', 'settings.yaml')
    # Use native path separators, with ./ prefix on Unix only
    rel_settings = os.path.relpath(settings_path)
    rel_data = os.path.relpath(os.path.join(target_dir, 'data')) + os.sep
    if os.sep == '/':
        rel_settings = './' + rel_settings
        rel_data = './' + rel_data
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f) or {}
                has_data_sources = bool(settings.get('data_sources'))
        except Exception:
            pass

    # Show next step
    print()
    print(f"Run {C.GREEN}tally workflow{C.RESET} to see next steps.")


def cmd_run(args):
    """Handle the 'run' subcommand."""
    # Determine config directory
    if args.config:
        config_dir = os.path.abspath(args.config)
    else:
        # Auto-detect config directory (supports both old and new layouts)
        config_dir = find_config_dir()

    if not config_dir or not os.path.isdir(config_dir):
        print(f"Error: Config directory not found.", file=sys.stderr)
        print(f"Looked for: ./config and ./tally/config", file=sys.stderr)
        print(f"\nRun 'tally init' to create a new budget directory.", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(config_dir, args.settings)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    year = config.get('year', 2025)
    home_locations = config.get('home_locations', set())
    travel_labels = config.get('travel_labels', {})
    data_sources = config.get('data_sources', [])
    cleaning_patterns = config.get('description_cleaning', [])

    # Check for data sources early before printing anything
    if not data_sources:
        print("Error: No data sources configured", file=sys.stderr)
        print(f"\nEdit {config_dir}/{args.settings} to add your data sources.", file=sys.stderr)
        print(f"\nExample:", file=sys.stderr)
        print(f"  data_sources:", file=sys.stderr)
        print(f"    - name: AMEX", file=sys.stderr)
        print(f"      file: data/amex.csv", file=sys.stderr)
        print(f"      type: amex", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Tally - {year}")
        print(f"Config: {config_dir}/{args.settings}")
        print()

    # Load merchant rules
    rules_file = os.path.join(config_dir, 'merchant_categories.csv')
    if os.path.exists(rules_file):
        rules = get_all_rules(rules_file)
        if not args.quiet:
            print(f"Loaded {len(rules)} categorization rules from {rules_file}")
            if len(rules) == 0:
                print()
                print("⚠️  No merchant rules defined - all transactions will be 'Unknown'")
                print("    Run 'tally discover' to find unknown merchants and get suggested rules.")
                print("    Tip: Use an AI agent with 'tally discover' to auto-generate rules!")
                print()
    else:
        rules = get_all_rules()  # No rules file
        if not args.quiet:
            print(f"No merchant_categories.csv found - transactions will be categorized as Unknown")

    # Parse transactions from configured data sources
    all_txns = []

    for source in data_sources:
        filepath = os.path.join(config_dir, '..', source['file'])
        filepath = os.path.normpath(filepath)

        if not os.path.exists(filepath):
            # Try relative to config_dir parent
            filepath = os.path.join(os.path.dirname(config_dir), source['file'])

        if not os.path.exists(filepath):
            if not args.quiet:
                print(f"  {source['name']}: File not found - {source['file']}")
            continue

        # Get parser type and format spec (set by config_loader.resolve_source_format)
        parser_type = source.get('_parser_type', source.get('type', '')).lower()
        format_spec = source.get('_format_spec')

        try:
            if parser_type == 'amex':
                txns = parse_amex(filepath, rules, home_locations, cleaning_patterns)
            elif parser_type == 'boa':
                txns = parse_boa(filepath, rules, home_locations, cleaning_patterns)
            elif parser_type == 'generic' and format_spec:
                txns = parse_generic_csv(filepath, format_spec, rules,
                                         home_locations,
                                         source_name=source.get('name', 'CSV'),
                                         decimal_separator=source.get('decimal_separator', '.'),
                                         cleaning_patterns=cleaning_patterns)
            else:
                if not args.quiet:
                    print(f"  {source['name']}: Unknown parser type '{parser_type}'")
                    print(f"    Use 'tally inspect {source['file']}' to determine format")
                continue
        except Exception as e:
            if not args.quiet:
                print(f"  {source['name']}: Error parsing - {e}")
            continue

        all_txns.extend(txns)
        if not args.quiet:
            print(f"  {source['name']}: {len(txns)} transactions")

    if not all_txns:
        print("Error: No transactions found", file=sys.stderr)
        sys.exit(1)

    # Auto-detect home location if not specified
    if not home_locations:
        from collections import Counter
        # US state codes for filtering
        us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR', 'VI', 'GU'
        }
        # Count US state locations
        location_counts = Counter(
            txn['location'] for txn in all_txns
            if txn.get('location') and txn['location'] in us_states
        )
        if location_counts:
            # Most common location is likely home
            detected_home = location_counts.most_common(1)[0][0]
            home_locations = {detected_home}
            if not args.quiet:
                print(f"Auto-detected home location: {detected_home}")
            # Update is_travel on transactions now that we know home
            from .analyzer import is_travel_location
            for txn in all_txns:
                txn['is_travel'] = is_travel_location(txn.get('location'), home_locations)

    if not args.quiet:
        print(f"\nTotal: {len(all_txns)} transactions")
        if home_locations:
            print(f"Home locations: {', '.join(sorted(home_locations))}")

    # Analyze
    stats = analyze_transactions(all_txns)

    # Parse filter options
    only_filter = None
    if args.only:
        valid_classifications = {'monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable'}
        only_filter = [c.strip() for c in args.only.split(',')]
        invalid = [c for c in only_filter if c not in valid_classifications]
        if invalid:
            print(f"Warning: Invalid classification(s) ignored: {', '.join(invalid)}", file=sys.stderr)
            print(f"  Valid options: {', '.join(sorted(valid_classifications))}", file=sys.stderr)
            only_filter = [c for c in only_filter if c in valid_classifications]
            if not only_filter:
                only_filter = None
    category_filter = args.category if hasattr(args, 'category') and args.category else None

    # Handle output format
    output_format = args.format if hasattr(args, 'format') else 'html'
    verbose = args.verbose if hasattr(args, 'verbose') else 0

    currency_format = config.get('currency_format', '${amount}')

    if output_format == 'json':
        # JSON output with reasoning
        from .analyzer import export_json
        print(export_json(stats, verbose=verbose, only=only_filter, category_filter=category_filter))
    elif output_format == 'markdown':
        # Markdown output with reasoning
        from .analyzer import export_markdown
        print(export_markdown(stats, verbose=verbose, only=only_filter, category_filter=category_filter))
    elif output_format == 'summary' or args.summary:
        # Text summary only (no HTML)
        print_summary(stats, year=year, currency_format=currency_format)
    else:
        # HTML output (default)
        # Print summary first
        if not args.quiet:
            print_summary(stats, year=year, currency_format=currency_format)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_dir = os.path.join(os.path.dirname(config_dir), config.get('output_dir', 'output'))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, config.get('html_filename', 'spending_summary.html'))

        # Collect source names for the report subtitle
        source_names = [s.get('name', 'Unknown') for s in data_sources]
        write_summary_file_vue(stats, output_path, year=year, home_locations=home_locations,
                               currency_format=currency_format, sources=source_names,
                               embedded_html=args.embedded_html)
        if not args.quiet:
            # Make the path clickable using OSC 8 hyperlink escape sequence
            abs_path = os.path.abspath(output_path)
            file_url = f"file://{abs_path}"
            # OSC 8 format: \033]8;;URL\033\\text\033]8;;\033\\
            clickable_path = f"\033]8;;{file_url}\033\\{output_path}\033]8;;\033\\"
            print(f"\nHTML report: {clickable_path}")


def cmd_discover(args):
    """Handle the 'discover' subcommand - find unknown merchants for rule creation."""
    from collections import Counter, defaultdict
    import re

    # Determine config directory
    if args.config:
        config_dir = os.path.abspath(args.config)
    else:
        config_dir = find_config_dir()

    if not config_dir or not os.path.isdir(config_dir):
        print(f"Error: Config directory not found.", file=sys.stderr)
        print(f"Looked for: ./config and ./tally/config", file=sys.stderr)
        print(f"\nRun 'tally init' to create a new budget directory.", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(config_dir, args.settings)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    home_locations = config.get('home_locations', set())
    data_sources = config.get('data_sources', [])
    cleaning_patterns = config.get('description_cleaning', [])

    if not data_sources:
        print("Error: No data sources configured", file=sys.stderr)
        print(f"\nEdit {config_dir}/{args.settings} to add your data sources.", file=sys.stderr)
        print(f"\nExample:", file=sys.stderr)
        print(f"  data_sources:", file=sys.stderr)
        print(f"    - name: AMEX", file=sys.stderr)
        print(f"      file: data/amex.csv", file=sys.stderr)
        print(f"      type: amex", file=sys.stderr)
        sys.exit(1)

    # Load merchant rules
    rules_file = os.path.join(config_dir, 'merchant_categories.csv')
    if os.path.exists(rules_file):
        rules = get_all_rules(rules_file)
    else:
        rules = get_all_rules()

    # Parse transactions from configured data sources
    all_txns = []

    for source in data_sources:
        filepath = os.path.join(config_dir, '..', source['file'])
        filepath = os.path.normpath(filepath)

        if not os.path.exists(filepath):
            filepath = os.path.join(os.path.dirname(config_dir), source['file'])

        if not os.path.exists(filepath):
            continue

        parser_type = source.get('_parser_type', source.get('type', '')).lower()
        format_spec = source.get('_format_spec')

        try:
            if parser_type == 'amex':
                txns = parse_amex(filepath, rules, home_locations, cleaning_patterns)
            elif parser_type == 'boa':
                txns = parse_boa(filepath, rules, home_locations, cleaning_patterns)
            elif parser_type == 'generic' and format_spec:
                txns = parse_generic_csv(filepath, format_spec, rules,
                                         home_locations,
                                         source_name=source.get('name', 'CSV'),
                                         decimal_separator=source.get('decimal_separator', '.'),
                                         cleaning_patterns=cleaning_patterns)
            else:
                continue
        except Exception:
            continue

        all_txns.extend(txns)

    if not all_txns:
        print("Error: No transactions found", file=sys.stderr)
        sys.exit(1)

    # Find unknown transactions
    unknown_txns = [t for t in all_txns if t.get('category') == 'Unknown']

    if not unknown_txns:
        print("No unknown transactions found! All merchants are categorized.")
        sys.exit(0)

    # Group by raw description and calculate stats
    desc_stats = defaultdict(lambda: {'count': 0, 'total': 0.0, 'examples': []})

    for txn in unknown_txns:
        raw = txn.get('raw_description', txn.get('description', ''))
        amount = abs(txn.get('amount', 0))
        desc_stats[raw]['count'] += 1
        desc_stats[raw]['total'] += amount
        if len(desc_stats[raw]['examples']) < 3:
            desc_stats[raw]['examples'].append(txn)

    # Sort by total spend (descending)
    sorted_descs = sorted(desc_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    # Limit output
    limit = args.limit
    if limit > 0:
        sorted_descs = sorted_descs[:limit]

    # Output format
    if args.format == 'csv':
        # CSV output for easy import
        print("# Suggested rules for unknown merchants")
        print("# Copy the lines you want to merchant_categories.csv")
        print("Pattern,Merchant,Category,Subcategory")
        print()

        for raw_desc, stats in sorted_descs:
            # Generate a suggested pattern
            pattern = suggest_pattern(raw_desc)
            # Generate a clean merchant name
            merchant = suggest_merchant_name(raw_desc)
            # Placeholder category - agent should fill in
            print(f"{pattern},{merchant},CATEGORY,SUBCATEGORY  # ${stats['total']:.2f} ({stats['count']} txns)")

    elif args.format == 'json':
        import json
        output = []
        for raw_desc, stats in sorted_descs:
            output.append({
                'raw_description': raw_desc,
                'suggested_pattern': suggest_pattern(raw_desc),
                'suggested_merchant': suggest_merchant_name(raw_desc),
                'count': stats['count'],
                'total_spend': round(stats['total'], 2),
                'examples': [
                    {
                        'date': str(t.get('date', '')),
                        'amount': t.get('amount', 0),
                        'description': t.get('description', '')
                    }
                    for t in stats['examples']
                ]
            })
        print(json.dumps(output, indent=2))

    else:
        # Default: human-readable format
        print(f"UNKNOWN MERCHANTS - Top {len(sorted_descs)} by spend")
        print("=" * 80)
        print(f"Total unknown: {len(unknown_txns)} transactions, ${sum(s['total'] for _, s in desc_stats.items()):.2f}")
        print()

        for i, (raw_desc, stats) in enumerate(sorted_descs, 1):
            pattern = suggest_pattern(raw_desc)
            merchant = suggest_merchant_name(raw_desc)

            print(f"{i}. {raw_desc[:60]}")
            print(f"   Count: {stats['count']} | Total: ${stats['total']:.2f}")
            print(f"   Suggested pattern: {pattern}")
            print(f"   Suggested merchant: {merchant}")
            print(f"   CSV: {pattern},{merchant},CATEGORY,SUBCATEGORY")
            print()


def suggest_pattern(description):
    """Generate a suggested regex pattern from a raw description."""
    import re

    desc = description.upper()

    # Remove common suffixes that vary
    desc = re.sub(r'\s+\d{4,}.*$', '', desc)  # Remove trailing numbers (store IDs)
    desc = re.sub(r'\s+[A-Z]{2}$', '', desc)  # Remove trailing state codes
    desc = re.sub(r'\s+\d{5}$', '', desc)  # Remove zip codes
    desc = re.sub(r'\s+#\d+', '', desc)  # Remove store numbers like #1234

    # Remove common prefixes
    prefixes = ['APLPAY ', 'SQ *', 'TST*', 'SP ', 'PP*', 'GOOGLE *']
    for prefix in prefixes:
        if desc.startswith(prefix):
            desc = desc[len(prefix):]

    # Clean up
    desc = desc.strip()

    # Escape regex special characters but keep it readable
    # Only escape characters that are common in descriptions
    pattern = re.sub(r'([.*+?^${}()|[\]\\])', r'\\\1', desc)

    # Simplify: take first 2-3 significant words
    words = pattern.split()[:3]
    if words:
        pattern = r'\s*'.join(words)

    return pattern


def suggest_merchant_name(description):
    """Generate a clean merchant name from a raw description."""
    import re

    desc = description

    # Remove common prefixes
    prefixes = ['APLPAY ', 'SQ *', 'TST*', 'TST* ', 'SP ', 'PP*', 'GOOGLE *']
    for prefix in prefixes:
        if desc.upper().startswith(prefix.upper()):
            desc = desc[len(prefix):]

    # Remove trailing IDs, numbers, locations
    desc = re.sub(r'\s+\d{4,}.*$', '', desc)
    desc = re.sub(r'\s+[A-Z]{2}$', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'\s+\d{5}$', '', desc)
    desc = re.sub(r'\s+#\d+', '', desc)
    desc = re.sub(r'\s+DES:.*$', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'\s+ID:.*$', '', desc, flags=re.IGNORECASE)

    # Take first few words and title case
    words = desc.split()[:3]
    if words:
        return ' '.join(words).title()

    return 'Unknown'


def cmd_inspect(args):
    """Handle the 'inspect' subcommand - show CSV structure and sample rows."""
    import csv

    if not args.file:
        print("Error: No file specified", file=sys.stderr)
        print("\nUsage: tally inspect <file.csv>", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  tally inspect data/transactions.csv", file=sys.stderr)
        sys.exit(1)

    filepath = os.path.abspath(args.file)

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    num_rows = args.rows

    print(f"Inspecting: {filepath}")
    print("=" * 70)

    with open(filepath, 'r', encoding='utf-8') as f:
        # Detect if it's a valid CSV
        try:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
            f.seek(0)
        except csv.Error:
            print("Warning: Could not detect CSV dialect, using default")
            dialect = None
            has_header = True
            f.seek(0)

        reader = csv.reader(f, dialect) if dialect else csv.reader(f)

        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= num_rows:  # Get header + N data rows
                break

        if not rows:
            print("File appears to be empty.")
            return

    # Display header info
    if has_header and rows:
        print("\nDetected Headers:")
        print("-" * 70)
        for idx, col in enumerate(rows[0]):
            print(f"  Column {idx}: {col}")

    # Display sample data
    print(f"\nSample Data (first {min(num_rows, len(rows)-1)} rows):")
    print("-" * 70)

    data_rows = rows[1:] if has_header else rows
    for row_num, row in enumerate(data_rows[:num_rows], start=1):
        print(f"\nRow {row_num}:")
        for idx, val in enumerate(row):
            header = rows[0][idx] if has_header and idx < len(rows[0]) else f"Col {idx}"
            # Truncate long values
            display_val = val[:50] + "..." if len(val) > 50 else val
            print(f"  [{idx}] {header}: {display_val}")

    # Attempt auto-detection
    print("\n" + "=" * 70)
    print("Auto-Detection Results:")
    print("-" * 70)

    try:
        spec = auto_detect_csv_format(filepath)
        print("  Successfully detected format!")
        print(f"  - Date column: {spec.date_column} (format: {spec.date_format})")
        print(f"  - Description column: {spec.description_column}")
        print(f"  - Amount column: {spec.amount_column}")
        if spec.location_column is not None:
            print(f"  - Location column: {spec.location_column}")

        # Build suggested format string
        max_col = max(spec.date_column, spec.description_column, spec.amount_column)
        if spec.location_column is not None:
            max_col = max(max_col, spec.location_column)

        cols = []
        for i in range(max_col + 1):
            if i == spec.date_column:
                cols.append(f'{{date:{spec.date_format}}}')
            elif i == spec.description_column:
                cols.append('{description}')
            elif i == spec.amount_column:
                cols.append('{amount}')
            elif spec.location_column is not None and i == spec.location_column:
                cols.append('{location}')
            else:
                cols.append('{_}')

        format_str = ', '.join(cols)
        print(f"\n  Suggested format string:")
        print(f'    format: "{format_str}"')

    except ValueError as e:
        print(f"  Could not auto-detect: {e}")
        print("\n  Use a manual format string. Example:")
        print('    format: "{date:%m/%d/%Y}, {description}, {amount}"')

    print()


def cmd_diag(args):
    """Handle the 'diag' subcommand - show diagnostic information about config and rules."""
    import json as json_module

    # Determine config directory
    if args.config:
        config_dir = os.path.abspath(args.config)
    else:
        config_dir = find_config_dir() or os.path.abspath('config')

    print("BUDGET ANALYZER DIAGNOSTICS")
    print("=" * 70)
    print()

    # Config directory info
    print("CONFIGURATION")
    print("-" * 70)
    print(f"Config directory: {config_dir}")
    print(f"  Exists: {os.path.isdir(config_dir)}")
    print()

    if not os.path.isdir(config_dir):
        print("ERROR: Config directory not found!")
        print("Run 'tally init' to create a new budget directory.")
        sys.exit(1)

    # Settings file
    settings_path = os.path.join(config_dir, args.settings)
    print(f"Settings file: {settings_path}")
    print(f"  Exists: {os.path.exists(settings_path)}")

    config = None
    if os.path.exists(settings_path):
        try:
            config = load_config(config_dir, args.settings)
            print(f"  Loaded successfully: Yes")
            print(f"  Year: {config.get('year', 'not set')}")
            print(f"  Output dir: {config.get('output_dir', 'not set')}")
            home_locs = config.get('home_locations', set())
            print(f"  Home locations: {', '.join(sorted(home_locs)) if home_locs else 'auto-detect'}")
            currency_fmt = config.get('currency_format', '${amount}')
            from .analyzer import format_currency
            print(f"  Currency format: {currency_fmt}")
            print(f"    Example: {format_currency(1234, currency_fmt)}")
            cleaning = config.get('description_cleaning', [])
            if cleaning:
                print(f"  Description cleaning: {len(cleaning)} pattern(s)")
                for pattern in cleaning:
                    print(f"    - {pattern}")
            else:
                print(f"  Description cleaning: none configured")
        except Exception as e:
            print(f"  Loaded successfully: No")
            print(f"  Error: {e}")
    print()

    # Data sources
    if config and config.get('data_sources'):
        print("DATA SOURCES")
        print("-" * 70)
        for i, source in enumerate(config['data_sources'], 1):
            filepath = os.path.join(config_dir, '..', source['file'])
            filepath = os.path.normpath(filepath)
            if not os.path.exists(filepath):
                filepath = os.path.join(os.path.dirname(config_dir), source['file'])

            print(f"  {i}. {source.get('name', 'unnamed')}")
            print(f"     File: {source['file']}")
            print(f"     Exists: {os.path.exists(filepath)}")
            if source.get('type'):
                print(f"     Type: {source['type']}")
            if source.get('format'):
                print(f"     Format: {source['format']}")

            # Show format spec details if available
            format_spec = source.get('_format_spec')
            if format_spec:
                print(f"     Columns:")
                print(f"       date: column {format_spec.date_column} (format: {format_spec.date_format})")
                print(f"       amount: column {format_spec.amount_column}")
                if format_spec.description_column is not None:
                    print(f"       description: column {format_spec.description_column}")
                if format_spec.custom_captures:
                    for name, col in format_spec.custom_captures.items():
                        print(f"       {name}: column {col} (custom capture)")
                if format_spec.description_template:
                    print(f"     Description template: {format_spec.description_template}")
                if format_spec.location_column is not None:
                    print(f"       location: column {format_spec.location_column}")
                if format_spec.negate_amount:
                    print(f"     Amount negation: enabled")
            print()

    # Merchant rules diagnostics
    print("MERCHANT RULES")
    print("-" * 70)

    rules_path = os.path.join(config_dir, 'merchant_categories.csv')
    diag = diagnose_rules(rules_path)

    print(f"User rules file: {diag['user_rules_path']}")
    print(f"  Exists: {diag['user_rules_exists']}")

    if diag['user_rules_exists']:
        print(f"  File size: {diag.get('file_size_bytes', 0)} bytes")
        print(f"  Total lines: {diag.get('file_lines', 0)}")
        print(f"  Non-comment lines: {diag.get('non_comment_lines', 0)}")
        print(f"  Has valid header: {diag.get('has_header', 'unknown')}")
        print(f"  Rules loaded: {diag['user_rules_count']}")

        if diag['user_rules_errors']:
            print()
            print("  ERRORS/WARNINGS:")
            for err in diag['user_rules_errors']:
                print(f"    - {err}")

        # Tag statistics
        if diag.get('rules_with_tags', 0) > 0:
            print()
            pct = (diag['rules_with_tags'] / diag['user_rules_count'] * 100) if diag['user_rules_count'] > 0 else 0
            print(f"  Rules with tags: {diag['rules_with_tags']}/{diag['user_rules_count']} ({pct:.0f}%)")
            if diag.get('unique_tags'):
                print(f"  Unique tags: {', '.join(sorted(diag['unique_tags']))}")

        if diag['user_rules']:
            print()
            print("  USER RULES (all):")
            for rule in diag['user_rules']:
                if len(rule) == 5:
                    pattern, merchant, category, subcategory, tags = rule
                else:
                    pattern, merchant, category, subcategory = rule
                    tags = []
                print(f"    {pattern}")
                tags_str = f" [{', '.join(tags)}]" if tags else ""
                print(f"      -> {merchant} | {category} > {subcategory}{tags_str}")
    else:
        print()
        print("  No merchant_categories.csv found.")
        print("  Transactions will be categorized as 'Unknown'.")
    print()

    print(f"Total rules: {diag['total_rules']}")
    print()

    # JSON output option
    if args.format == 'json':
        print("JSON OUTPUT")
        print("-" * 70)
        output = {
            'config_dir': config_dir,
            'config_dir_exists': os.path.isdir(config_dir),
            'settings_file': settings_path,
            'settings_exists': os.path.exists(settings_path),
            'data_sources': [],
            'rules': {
                'user_rules_path': diag['user_rules_path'],
                'user_rules_exists': diag['user_rules_exists'],
                'user_rules_count': diag['user_rules_count'],
                'user_rules': [
                    {'pattern': r[0], 'merchant': r[1], 'category': r[2], 'subcategory': r[3], 'tags': r[4] if len(r) > 4 else []}
                    for r in diag['user_rules']
                ],
                'errors': diag['user_rules_errors'],
                'total_rules': diag['total_rules'],
                'rules_with_tags': diag.get('rules_with_tags', 0),
                'unique_tags': sorted(diag.get('unique_tags', set())),
            }
        }
        if config and config.get('data_sources'):
            for source in config['data_sources']:
                filepath = os.path.join(os.path.dirname(config_dir), source['file'])
                output['data_sources'].append({
                    'name': source.get('name'),
                    'file': source['file'],
                    'exists': os.path.exists(filepath),
                    'type': source.get('type'),
                    'format': source.get('format'),
                })
        print(json_module.dumps(output, indent=2))


def cmd_workflow(args):
    """Show context-aware workflow instructions for AI agents."""
    import subprocess

    # Detect current state
    config_dir = find_config_dir()
    has_config = config_dir is not None
    has_data_sources = False
    unknown_count = 0
    total_unknown_spend = 0

    # Calculate relative paths for display (OS-aware)
    def make_path(relative_to_config_parent, trailing_sep=False):
        """Create display path relative to cwd with correct OS separators."""
        if config_dir:
            parent = os.path.dirname(config_dir)
            full_path = os.path.join(parent, relative_to_config_parent)
        else:
            full_path = relative_to_config_parent
        rel = os.path.relpath(full_path)
        if trailing_sep:
            rel = rel + os.sep
        # Add ./ prefix on Unix only
        if os.sep == '/' and not rel.startswith('.'):
            rel = './' + rel
        return rel

    # Default paths (used when no config exists)
    path_data = make_path('data', trailing_sep=True) if config_dir else './data/'
    path_settings = make_path(os.path.join('config', 'settings.yaml')) if config_dir else './config/settings.yaml'
    path_rules = make_path(os.path.join('config', 'merchant_categories.csv')) if config_dir else './config/merchant_categories.csv'

    if has_config:
        try:
            config = load_config(config_dir)
            has_data_sources = bool(config.get('data_sources'))

            if has_data_sources:
                # Try to get unknown merchant count
                try:
                    result = subprocess.run(
                        ['tally', 'discover', '--format', 'json'],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        import json as json_module
                        unknowns = json_module.loads(result.stdout)
                        unknown_count = len(unknowns)
                        total_unknown_spend = sum(u.get('total_spend', 0) for u in unknowns)
                except Exception:
                    pass
        except Exception:
            pass

    # Helper for section headers
    def section(title):
        print()
        print(f"{C.BOLD}{C.CYAN}▸ {title}{C.RESET}")

    # Build context-aware output
    print()
    print(f"{C.BOLD}  TALLY WORKFLOW{C.RESET}")
    print(f"{C.DIM}  ─────────────────────────────────────────{C.RESET}")

    # Status bar
    if not has_config:
        print(f"  {C.YELLOW}●{C.RESET} No config found")
        section("Getting Started")
        print(f"    {C.DIM}1.{C.RESET} Initialize the project:")
        print(f"       {C.GREEN}tally init{C.RESET}")
        print()
        print(f"    {C.DIM}2.{C.RESET} Add bank/credit card CSVs to {C.CYAN}./data/{C.RESET}")
        print()
        print(f"    {C.DIM}3.{C.RESET} Configure data sources in {C.CYAN}./config/settings.yaml{C.RESET}")
        print()
        return

    if not has_data_sources:
        print(f"  {C.YELLOW}●{C.RESET} No data sources configured")
        section("Setup Data Sources")
        print(f"    {C.DIM}1.{C.RESET} Add bank/credit card CSVs to {C.CYAN}{path_data}{C.RESET}")
        print()
        print(f"    {C.DIM}2.{C.RESET} Inspect your file to get the format string:")
        print(f"       {C.GREEN}tally inspect {path_data}yourfile.csv{C.RESET}")
        print()
        print(f"    {C.DIM}3.{C.RESET} Add to {C.CYAN}{path_settings}{C.RESET}:")
        print(f"       {C.DIM}data_sources:")
        print(f"         - name: My Card")
        print(f"           file: data/transactions.csv")
        print(f"           format: \"{{date:%m/%d/%Y}},{{description}},{{amount}}\"{C.RESET}")
        print()
        return

    # Configured state
    if unknown_count > 0:
        print(f"  {C.GREEN}●{C.RESET} Config ready  {C.DIM}│{C.RESET}  {C.YELLOW}●{C.RESET} {unknown_count} unknown merchants {C.DIM}(${total_unknown_spend:,.0f}){C.RESET}")
    else:
        print(f"  {C.GREEN}●{C.RESET} Config ready  {C.DIM}│{C.RESET}  {C.GREEN}●{C.RESET} All merchants categorized")

    # Show categorization workflow if there are unknowns
    if unknown_count > 0:
        section("Categorization Workflow")
        print(f"    {C.DIM}1.{C.RESET} Get unknown merchants with suggested patterns:")
        print(f"       {C.GREEN}tally discover --format json{C.RESET}")
        print()
        print(f"    {C.DIM}2.{C.RESET} Add rules to {C.CYAN}{path_rules}{C.RESET}:")
        print(f"       {C.DIM}Pattern,Merchant,Category,Subcategory,Tags")
        print(f"       STARBUCKS,Starbucks,Food,Coffee,")
        print(f"       UBER\\s(?!EATS),Uber,Transport,Rideshare,business{C.RESET}")
        print()
        print(f"    {C.DIM}3.{C.RESET} Check progress:")
        print(f"       {C.GREEN}tally run --summary{C.RESET}")
        print()
        print(f"    {C.DIM}Repeat until all merchants are categorized{C.RESET}")

    section("Commands")
    cmds = [
        ("tally run", "Generate HTML spending report"),
        ("tally run --summary", "Quick text summary"),
        ("tally discover", "Find unknown merchants"),
        ("tally explain <merchant>", "Debug classification"),
        ("tally diag", "Diagnose config issues"),
    ]
    for cmd, desc in cmds:
        print(f"    {C.GREEN}{cmd:<24}{C.RESET} {C.DIM}{desc}{C.RESET}")

    section("Debugging Rules")
    print(f"    {C.GREEN}tally explain Amazon{C.RESET}           {C.DIM}See how Amazon is classified{C.RESET}")
    print(f"    {C.GREEN}tally explain Amazon -v{C.RESET}        {C.DIM}Show raw description variations{C.RESET}")
    print(f"    {C.GREEN}tally explain \"RAW DESC\"{C.RESET}       {C.DIM}Trace how a description matches{C.RESET}")

    section("Description Cleaning")
    print(f"    {C.DIM}Strip payment processor prefixes before matching rules.{C.RESET}")
    print(f"    {C.DIM}Add to {C.RESET}{C.CYAN}{path_settings}{C.RESET}{C.DIM}:{C.RESET}")
    print()
    print(f"    {C.DIM}description_cleaning:")
    print(f"      - \"^APLPAY\\\\s+\"       # Apple Pay")
    print(f"      - \"^SQ\\\\s*\\\\*\"       # Square")
    print(f"      - \"\\\\s+DES:.*$\"      # BOA suffix{C.RESET}")

    section("CSV Format")
    print(f"    {C.DIM}Pattern,Merchant,Category,Subcategory,Tags{C.RESET}")
    print()
    print(f"    {C.BOLD}Pattern{C.RESET}      Python regex, case-insensitive")
    print(f"    {C.BOLD}Tags{C.RESET}         Optional, pipe-separated: {C.DIM}business|reimbursable{C.RESET}")
    print()
    print(f"    {C.DIM}Examples:{C.RESET}")
    patterns = [
        ("STARBUCKS", "contains STARBUCKS"),
        ("UBER|LYFT", "either UBER or LYFT"),
        ("UBER\\s(?!EATS)", "UBER but not UBER EATS"),
        ("COSTCO[amount>200]", "Costco orders over $200"),
    ]
    for pattern, desc in patterns:
        print(f"      {C.CYAN}{pattern:<22}{C.RESET} {C.DIM}{desc}{C.RESET}")

    print()
    print(f"  {C.DIM}First match wins — put specific patterns before general ones{C.RESET}")
    print()


def cmd_update(args):
    """Handle the update command."""
    if args.prerelease:
        print("Checking for development builds...")
    else:
        print("Checking for updates...")

    # Get release info (may fail if offline or rate-limited)
    release_info = get_latest_release_info(prerelease=args.prerelease)
    has_update = False

    if release_info:
        latest = release_info['version']
        current = VERSION

        # Show version comparison
        from ._version import _version_greater
        has_update = _version_greater(latest, current)

        if has_update:
            if args.prerelease:
                print(f"Development build available: v{latest} (current: v{current})")
            else:
                print(f"New version available: v{latest} (current: v{current})")
        else:
            print(f"Already on latest version: v{current}")
    else:
        if args.prerelease:
            print("No development build found. Dev builds are created on each push to main.")
        else:
            print("Could not check for version updates (network issue?)")

    # If --check only, just show status and exit
    if args.check:
        if has_update:
            if args.prerelease:
                print(f"\nRun 'tally update --prerelease' to install the development build.")
            else:
                print(f"\nRun 'tally update' to install the update.")
        sys.exit(0)

    # Check for migrations (layout updates, etc.)
    # This runs even if version check failed
    config_dir = find_config_dir()
    did_migrate = False
    if config_dir:
        old_config = config_dir
        new_config = run_migrations(config_dir, skip_confirm=args.yes)
        if new_config and new_config != old_config:
            did_migrate = True

    # Skip binary update if no update available
    if not has_update:
        if not did_migrate:
            print("\nNothing to update.")
        sys.exit(0)

    # Check if running from source (can't self-update)
    import sys as _sys
    if not getattr(_sys, 'frozen', False):
        print(f"\n✗ Cannot self-update when running from source. Use: uv tool upgrade tally")
        sys.exit(1)

    # Perform binary update
    print()
    success, message = perform_update(release_info)

    if success:
        print(f"\n✓ {message}")
        print("\nRestart tally to use the new version.")
    else:
        print(f"\n✗ {message}")
        sys.exit(1)


def cmd_explain(args):
    """Handle the 'explain' subcommand - explain merchant classifications."""
    from difflib import get_close_matches
    from .analyzer import export_json, export_markdown, build_merchant_json

    # Determine config directory
    # Check if first merchant arg looks like a config path
    config_dir = None
    merchant_names = args.merchant if args.merchant else []

    if merchant_names and os.path.isdir(merchant_names[-1]):
        # Last arg is a directory, treat it as config
        config_dir = os.path.abspath(merchant_names[-1])
        merchant_names = merchant_names[:-1]
    elif args.config:
        config_dir = os.path.abspath(args.config)
    else:
        config_dir = find_config_dir()

    if not config_dir or not os.path.isdir(config_dir):
        print(f"Error: Config directory not found.", file=sys.stderr)
        print(f"Looked for: ./config and ./tally/config", file=sys.stderr)
        print(f"\nRun 'tally init' to create a new budget directory.", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(config_dir, args.settings)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    home_locations = config.get('home_locations', set())
    data_sources = config.get('data_sources', [])
    cleaning_patterns = config.get('description_cleaning', [])

    if not data_sources:
        print("Error: No data sources configured", file=sys.stderr)
        sys.exit(1)

    # Load merchant rules
    rules_file = os.path.join(config_dir, 'merchant_categories.csv')
    if os.path.exists(rules_file):
        rules = get_all_rules(rules_file)
    else:
        rules = get_all_rules()

    # Parse transactions (quietly)
    all_txns = []
    for source in data_sources:
        filepath = os.path.join(config_dir, '..', source['file'])
        filepath = os.path.normpath(filepath)
        if not os.path.exists(filepath):
            filepath = os.path.join(os.path.dirname(config_dir), source['file'])
        if not os.path.exists(filepath):
            continue

        parser_type = source.get('_parser_type', source.get('type', '')).lower()
        format_spec = source.get('_format_spec')

        try:
            if parser_type == 'amex':
                txns = parse_amex(filepath, rules, home_locations, cleaning_patterns)
            elif parser_type == 'boa':
                txns = parse_boa(filepath, rules, home_locations, cleaning_patterns)
            elif parser_type == 'generic' and format_spec:
                txns = parse_generic_csv(filepath, format_spec, rules,
                                         home_locations,
                                         source_name=source.get('name', 'CSV'),
                                         decimal_separator=source.get('decimal_separator', '.'),
                                         cleaning_patterns=cleaning_patterns)
            else:
                continue
        except Exception:
            continue

        all_txns.extend(txns)

    if not all_txns:
        print("Error: No transactions found", file=sys.stderr)
        sys.exit(1)

    # Analyze
    stats = analyze_transactions(all_txns)

    # Get all merchants from all classifications
    all_merchants = {}
    for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
        merchants_dict = stats.get(f'{section}_merchants', {})
        for name, data in merchants_dict.items():
            all_merchants[name] = data

    verbose = args.verbose

    # Handle output based on what was requested
    if merchant_names:
        # Explain specific merchants
        found_any = False
        for merchant_query in merchant_names:
            # Try exact match first
            if merchant_query in all_merchants:
                found_any = True
                _print_merchant_explanation(merchant_query, all_merchants[merchant_query], args.format, verbose, stats['num_months'])
            else:
                # Try case-insensitive match
                matches = [m for m in all_merchants.keys() if m.lower() == merchant_query.lower()]
                if matches:
                    found_any = True
                    _print_merchant_explanation(matches[0], all_merchants[matches[0]], args.format, verbose, stats['num_months'])
                else:
                    # Try treating query as a raw description
                    trace = explain_description(merchant_query, rules, cleaning_patterns=cleaning_patterns)
                    if not trace['is_unknown']:
                        # It matched a rule - show the explanation
                        found_any = True
                        _print_description_explanation(merchant_query, trace, args.format, verbose)
                    else:
                        # Try fuzzy match on merchant names
                        close_matches = get_close_matches(merchant_query, list(all_merchants.keys()), n=3, cutoff=0.6)
                        if close_matches:
                            print(f"No merchant matching '{merchant_query}'. Did you mean:", file=sys.stderr)
                            for m in close_matches:
                                print(f"  - {m}", file=sys.stderr)
                        else:
                            # Show unknown merchant info
                            _print_description_explanation(merchant_query, trace, args.format, verbose)

        if not found_any:
            sys.exit(1)

    elif args.classification:
        # Show all merchants in a specific classification
        section = args.classification
        merchants_dict = stats.get(f'{section}_merchants', {})
        if not merchants_dict:
            print(f"No merchants in classification '{section}'")
            sys.exit(0)

        if args.format == 'json':
            import json
            merchants = [build_merchant_json(name, data, verbose) for name, data in merchants_dict.items()]
            merchants.sort(key=lambda x: x['monthly_value'], reverse=True)
            print(json.dumps({'classification': section, 'merchants': merchants}, indent=2))
        elif args.format == 'markdown':
            print(export_markdown(stats, verbose=verbose, only=[section], category_filter=args.category))
        else:
            # Text format
            _print_classification_summary(section, merchants_dict, verbose, stats['num_months'])

    elif args.category:
        # Filter by category across all classifications
        if args.format == 'json':
            print(export_json(stats, verbose=verbose, category_filter=args.category))
        elif args.format == 'markdown':
            print(export_markdown(stats, verbose=verbose, category_filter=args.category))
        else:
            # Text format - show all merchants in category
            print(f"Merchants in category: {args.category}\n")
            found_any = False
            for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
                merchants_dict = stats.get(f'{section}_merchants', {})
                section_merchants = {k: v for k, v in merchants_dict.items() if v.get('category') == args.category}
                if section_merchants:
                    found_any = True
                    _print_classification_summary(section, section_merchants, verbose, stats['num_months'])
            if not found_any:
                # Suggest categories that do exist
                all_categories = set()
                for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
                    for data in stats.get(f'{section}_merchants', {}).values():
                        if data.get('category'):
                            all_categories.add(data['category'])
                print(f"No merchants found in category '{args.category}'")
                if all_categories:
                    print(f"\nAvailable categories: {', '.join(sorted(all_categories))}")

    elif hasattr(args, 'tags') and args.tags:
        # Filter by tags across all classifications
        filter_tags = set(t.strip().lower() for t in args.tags.split(','))

        if args.format == 'json':
            # Filter merchants by tags and output JSON
            import json
            from .analyzer import build_merchant_json
            matched_merchants = []
            for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
                for name, data in stats.get(f'{section}_merchants', {}).items():
                    merchant_tags = set(t.lower() for t in data.get('tags', []))
                    if merchant_tags & filter_tags:
                        matched_merchants.append(build_merchant_json(name, data, verbose))
            matched_merchants.sort(key=lambda x: x['monthly_value'], reverse=True)
            print(json.dumps({'tags': list(filter_tags), 'merchants': matched_merchants}, indent=2))
        else:
            # Text format - show all merchants with tags
            print(f"Merchants with tags: {', '.join(sorted(filter_tags))}\n")
            found_any = False
            for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
                merchants_dict = stats.get(f'{section}_merchants', {})
                section_merchants = {
                    k: v for k, v in merchants_dict.items()
                    if set(t.lower() for t in v.get('tags', [])) & filter_tags
                }
                if section_merchants:
                    found_any = True
                    _print_classification_summary(section, section_merchants, verbose, stats['num_months'])
            if not found_any:
                # Suggest tags that do exist
                all_tags = set()
                for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
                    for data in stats.get(f'{section}_merchants', {}).values():
                        all_tags.update(data.get('tags', []))
                print(f"No merchants found with tags: {', '.join(sorted(filter_tags))}")
                if all_tags:
                    print(f"\nAvailable tags: {', '.join(sorted(all_tags))}")

    else:
        # No specific merchant - show classification summary
        _print_explain_summary(stats, verbose)


def _print_description_explanation(query, trace, output_format, verbose):
    """Print explanation for how a raw description matches."""
    import json

    if output_format == 'json':
        print(json.dumps(trace, indent=2))
    elif output_format == 'markdown':
        print(f"## Description Trace: `{query}`")
        print()
        if trace['cleaned'] and trace['cleaned'] != trace['original']:
            print(f"**Cleaned:** `{trace['cleaned']}`")
            print()

        if trace['is_unknown']:
            print(f"**Result:** Unknown merchant")
            print(f"**Extracted Name:** {trace['merchant']}")
            print()
            print("No matching rule found. Run `tally discover` to add a rule for this merchant.")
        else:
            rule = trace['matched_rule']
            print(f"**Matched Rule:** `{rule['pattern']}`")
            print(f"**Matched On:** {rule['matched_on']} description")
            print(f"**Merchant:** {trace['merchant']}")
            print(f"**Category:** {trace['category']} > {trace['subcategory']}")
            if rule.get('tags'):
                print(f"**Tags:** {', '.join(rule['tags'])}")
        print()
    else:
        # Text format
        print(f"Description: {query}")
        if trace['cleaned'] and trace['cleaned'] != trace['original']:
            print(f"  Cleaned: {trace['cleaned']}")

        print()
        if trace['is_unknown']:
            print(f"  Result: Unknown merchant")
            print(f"  Extracted name: {trace['merchant']}")
            print()
            print("  No matching rule found.")
            print("  Run 'tally discover' to add a rule for this merchant.")
        else:
            rule = trace['matched_rule']
            print(f"  Matched: {rule['pattern']} -> {trace['merchant']}")
            print(f"  Category: {trace['category']} > {trace['subcategory']}")
            if rule.get('tags'):
                print(f"  Tags: {', '.join(rule['tags'])}")
            if verbose >= 1:
                print(f"  Matched on: {rule['matched_on']} description")
        print()


def _print_merchant_explanation(name, data, output_format, verbose, num_months):
    """Print explanation for a single merchant."""
    import json
    from .analyzer import build_merchant_json

    if output_format == 'json':
        print(json.dumps(build_merchant_json(name, data, verbose), indent=2))
    elif output_format == 'markdown':
        reasoning = data.get('reasoning', {})
        print(f"## {name}")
        print(f"**Classification:** {data.get('classification', 'unknown').replace('_', ' ').title()}")
        print(f"**Reason:** {reasoning.get('decision', 'N/A')}")
        print(f"**Category:** {data.get('category', '')} > {data.get('subcategory', '')}")
        print(f"**Monthly Value:** ${data.get('monthly_value', 0):.2f}")
        print(f"**YTD Total:** ${data.get('total', 0):.2f}")
        print(f"**Months Active:** {data.get('months_active', 0)}/{num_months}")

        if verbose >= 1:
            # Show raw description variations
            raw_descs = data.get('raw_descriptions', {})
            if raw_descs and len(raw_descs) > 0:
                sorted_descs = sorted(raw_descs.items(), key=lambda x: -x[1])
                if verbose >= 2:
                    # -vv: show all variations
                    print(f"\n**Description Variations ({len(raw_descs)}):**")
                    for desc, count in sorted_descs:
                        print(f"  - `{desc}` ({count})")
                else:
                    # -v: show top 10 variations
                    print(f"\n**Description Variations ({len(raw_descs)} unique):**")
                    for desc, count in sorted_descs[:10]:
                        print(f"  - `{desc}` ({count})")
                    if len(raw_descs) > 10:
                        print(f"  - ... and {len(raw_descs) - 10} more (use -vv to see all)")

            trace = reasoning.get('trace', [])
            if trace:
                print('\n**Decision Trace:**')
                for i, step in enumerate(trace, 1):
                    print(f"  {i}. {step}")

        if verbose >= 2:
            print(f"\n**Calculation:** {data.get('calc_type', '')} ({data.get('calc_reasoning', '')})")
            print(f"  Formula: {data.get('calc_formula', '')}")

        # Show tags
        tags = data.get('tags', [])
        if tags:
            print(f"**Tags:** {', '.join(sorted(tags))}")

        # Show pattern match info
        match_info = data.get('match_info')
        if match_info:
            pattern = match_info.get('pattern', '')
            source = match_info.get('source', 'unknown')
            print(f"\n**Pattern:** `{pattern}` ({source})")
        print()
    else:
        # Text format
        classification = data.get('classification', 'unknown').replace('_', ' ').title()
        reasoning = data.get('reasoning', {})
        print(f"{name} -> {classification}")
        print(f"  {reasoning.get('decision', 'N/A')}")

        # Show tags
        tags = data.get('tags', [])
        if tags:
            print(f"  Tags: {', '.join(sorted(tags))}")

        if verbose >= 1:
            # Show raw description variations
            raw_descs = data.get('raw_descriptions', {})
            if raw_descs and len(raw_descs) > 0:
                sorted_descs = sorted(raw_descs.items(), key=lambda x: -x[1])
                if verbose >= 2:
                    # -vv: show all variations
                    print()
                    print(f"  Description variations ({len(raw_descs)}):")
                    for desc, count in sorted_descs:
                        print(f"    {desc} ({count})")
                else:
                    # -v: show top 10 variations
                    print()
                    print(f"  Description variations ({len(raw_descs)} unique):")
                    for desc, count in sorted_descs[:10]:
                        print(f"    {desc} ({count})")
                    if len(raw_descs) > 10:
                        print(f"    ... and {len(raw_descs) - 10} more (use -vv to see all)")

            trace = reasoning.get('trace', [])
            if trace:
                print()
                print("  Decision trace:")
                for step in trace:
                    print(f"    {step}")

        if verbose >= 2:
            print()
            print(f"  Calculation: {data.get('calc_type', '')} ({data.get('calc_reasoning', '')})")
            print(f"    Formula: {data.get('calc_formula', '')}")
            print(f"    CV: {reasoning.get('cv', 0):.2f}")

        # Show pattern match info (always show if available)
        match_info = data.get('match_info')
        if match_info:
            pattern = match_info.get('pattern', '')
            source = match_info.get('source', 'unknown')
            print(f"\n  Rule: {pattern} ({source})")
        print()


def _print_classification_summary(section, merchants_dict, verbose, num_months):
    """Print summary of merchants in a classification."""
    section_name = section.replace('_', ' ').title()
    print(f"{section_name} ({len(merchants_dict)} merchants)")
    print("-" * 50)

    sorted_merchants = sorted(merchants_dict.items(), key=lambda x: x[1].get('monthly_value', 0), reverse=True)
    for name, data in sorted_merchants:
        reasoning = data.get('reasoning', {})
        category = data.get('category', '')
        months = data.get('months_active', 0)

        # Short reason
        decision = reasoning.get('decision', '')
        short_reason = f"{category} ({months}/{num_months} months)"

        print(f"  {name:<24} {short_reason}")

        if verbose >= 1:
            trace = reasoning.get('trace', [])
            if trace:
                for step in trace:
                    print(f"    {step}")
            print()

    print()


def _print_explain_summary(stats, verbose):
    """Print overview summary of all classifications with brief reasons."""
    section_names = {
        'monthly': 'Monthly Recurring',
        'annual': 'Annual Bills',
        'periodic': 'Periodic Recurring',
        'travel': 'Travel',
        'one_off': 'One-Off',
        'variable': 'Variable/Discretionary',
    }

    print("Classification Summary")
    print("=" * 60)
    print()

    num_months = stats['num_months']

    for section in ['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable']:
        merchants_dict = stats.get(f'{section}_merchants', {})
        if not merchants_dict:
            continue

        section_name = section_names.get(section, section)
        print(f"{section_name} ({len(merchants_dict)} merchants)")

        sorted_merchants = sorted(merchants_dict.items(), key=lambda x: x[1].get('monthly_value', 0), reverse=True)

        # Show top 5 or all if verbose
        display_count = len(sorted_merchants) if verbose >= 1 else min(5, len(sorted_merchants))

        for name, data in sorted_merchants[:display_count]:
            category = data.get('category', '')
            months = data.get('months_active', 0)
            cv = data.get('cv', 0)

            # Short classification hint
            if data.get('is_consistent', True):
                consistency = "consistent"
            else:
                consistency = "varies"

            print(f"  {name:<26} {category} ({months}/{num_months} months, {consistency})")

        if len(sorted_merchants) > display_count:
            remaining = len(sorted_merchants) - display_count
            print(f"  ... and {remaining} more")

        print()

    print("Run `tally explain <merchant>` for detailed reasoning.")
    print("Run `tally explain -v` for full details on all merchants.")


def main():
    """Main entry point for tally CLI."""
    parser = argparse.ArgumentParser(
        prog='tally',
        description='A tool to help agents classify your bank transactions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Run 'tally workflow' to see next steps based on your current state.'''
    )

    subparsers = parser.add_subparsers(dest='command', title='commands', metavar='<command>')

    # init subcommand
    init_parser = subparsers.add_parser(
        'init',
        help='Set up a new budget folder with config files (run once to get started)'
    )
    init_parser.add_argument(
        'dir',
        nargs='?',
        default='tally',
        help='Directory to initialize (default: ./tally)'
    )

    # run subcommand
    run_parser = subparsers.add_parser(
        'run',
        help='Parse transactions, categorize them, and generate HTML spending report'
    )
    run_parser.add_argument(
        'config',
        nargs='?',
        help='Path to config directory (default: ./config)'
    )
    run_parser.add_argument(
        '--settings', '-s',
        default='settings.yaml',
        help='Settings file name (default: settings.yaml)'
    )
    run_parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary only, do not generate HTML'
    )
    run_parser.add_argument(
        '--output', '-o',
        help='Override output file path'
    )
    run_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    run_parser.add_argument(
        '--format', '-f',
        choices=['html', 'json', 'markdown', 'summary'],
        default='html',
        help='Output format: html (default), json (with reasoning), markdown, summary (text)'
    )
    run_parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase output verbosity (use -v for trace, -vv for full details)'
    )
    run_parser.add_argument(
        '--only',
        help='Filter to specific classifications (comma-separated: monthly,variable,travel)'
    )
    run_parser.add_argument(
        '--category',
        help='Filter to specific category'
    )
    run_parser.add_argument(
        '--tags',
        help='Filter by tags (comma-separated, e.g., --tags business,reimbursable)'
    )
    run_parser.add_argument(
        '--no-embedded-html',
        dest='embedded_html',
        action='store_false',
        default=True,
        help='Output CSS/JS as separate files instead of embedding (easier to iterate on styling)'
    )

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Show CSV columns and sample data to help build a format string',
        description='Show headers and sample rows from a CSV file, with auto-detection suggestions.'
    )
    inspect_parser.add_argument(
        'file',
        nargs='?',
        help='Path to the CSV file to inspect'
    )
    inspect_parser.add_argument(
        '--rows', '-n',
        type=int,
        default=5,
        help='Number of sample rows to display (default: 5)'
    )

    # discover subcommand
    discover_parser = subparsers.add_parser(
        'discover',
        help='List uncategorized transactions with suggested patterns (use --format json for LLMs)',
        description='Analyze transactions to find unknown merchants, sorted by spend. '
                    'Outputs suggested patterns for merchant_categories.csv.'
    )
    discover_parser.add_argument(
        'config',
        nargs='?',
        help='Path to config directory (default: ./config)'
    )
    discover_parser.add_argument(
        '--settings', '-s',
        default='settings.yaml',
        help='Settings file name (default: settings.yaml)'
    )
    discover_parser.add_argument(
        '--limit', '-n',
        type=int,
        default=20,
        help='Maximum number of unknown merchants to show (default: 20, 0 for all)'
    )
    discover_parser.add_argument(
        '--format', '-f',
        choices=['text', 'csv', 'json'],
        default='text',
        help='Output format: text (human readable), csv (for import), json (for agents)'
    )

    # diag subcommand
    diag_parser = subparsers.add_parser(
        'diag',
        help='Debug config issues: show loaded rules, data sources, and errors',
        description='Display detailed diagnostic info to help troubleshoot rule loading issues.'
    )
    diag_parser.add_argument(
        'config',
        nargs='?',
        help='Path to config directory (default: ./config)'
    )
    diag_parser.add_argument(
        '--settings', '-s',
        default='settings.yaml',
        help='Settings file name (default: settings.yaml)'
    )
    diag_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format: text (human readable), json (for agents)'
    )

    # explain subcommand
    explain_parser = subparsers.add_parser(
        'explain',
        help='Explain why merchants are classified the way they are',
        description='Show classification reasoning for merchants. '
                    'Runs analysis on-the-fly and explains the decision process.'
    )
    explain_parser.add_argument(
        'merchant',
        nargs='*',
        help='Merchant name(s) to explain (optional, shows summary if omitted)'
    )
    explain_parser.add_argument(
        'config',
        nargs='?',
        help='Path to config directory (default: ./config)'
    )
    explain_parser.add_argument(
        '--settings', '-s',
        default='settings.yaml',
        help='Settings file name (default: settings.yaml)'
    )
    explain_parser.add_argument(
        '--format', '-f',
        choices=['text', 'json', 'markdown'],
        default='text',
        help='Output format: text (default), json, markdown'
    )
    explain_parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase output verbosity (use -v for trace, -vv for full details)'
    )
    explain_parser.add_argument(
        '--classification', '-c',
        choices=['monthly', 'annual', 'periodic', 'travel', 'one_off', 'variable'],
        help='Show all merchants in a specific classification'
    )
    explain_parser.add_argument(
        '--category',
        help='Filter to specific category'
    )
    explain_parser.add_argument(
        '--tags',
        help='Filter by tags (comma-separated, e.g., --tags business,reimbursable)'
    )

    # workflow subcommand
    subparsers.add_parser(
        'workflow',
        help='Show context-aware workflow instructions for AI agents',
        description='Detects current state and shows relevant next steps.'
    )

    # version subcommand
    subparsers.add_parser(
        'version',
        help='Show version information',
        description='Display tally version and build information.'
    )

    # update subcommand
    update_parser = subparsers.add_parser(
        'update',
        help='Update tally to the latest version',
        description='Download and install the latest tally release.'
    )
    update_parser.add_argument(
        '--check',
        action='store_true',
        help='Check for updates without installing'
    )
    update_parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompts'
    )
    update_parser.add_argument(
        '--prerelease',
        action='store_true',
        help='Install latest development build from main branch'
    )

    args = parser.parse_args()

    # If no command specified, show help with banner
    if args.command is None:
        print(BANNER)
        parser.print_help()

        # Check for updates
        update_info = check_for_updates()
        if update_info and update_info.get('update_available'):
            print()
            if update_info.get('is_prerelease'):
                print(f"Dev build available: v{update_info['latest_version']} (current: v{update_info['current_version']})")
                print(f"  Run 'tally update --prerelease' to install")
            else:
                print(f"Update available: v{update_info['latest_version']} (current: v{update_info['current_version']})")
                print(f"  Run 'tally update' to install")

        sys.exit(0)

    # Dispatch to command handler
    if args.command == 'init':
        cmd_init(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'inspect':
        cmd_inspect(args)
    elif args.command == 'discover':
        cmd_discover(args)
    elif args.command == 'diag':
        cmd_diag(args)
    elif args.command == 'explain':
        cmd_explain(args)
    elif args.command == 'workflow':
        cmd_workflow(args)
    elif args.command == 'version':
        sha_display = GIT_SHA[:8] if GIT_SHA != 'unknown' else 'unknown'
        print(f"tally {VERSION} ({sha_display})")
        print(REPO_URL)

        # Check for updates
        update_info = check_for_updates()
        if update_info and update_info.get('update_available'):
            print()
            if update_info.get('is_prerelease'):
                print(f"Dev build available: v{update_info['latest_version']}")
                print(f"  Run 'tally update --prerelease' to install")
            else:
                print(f"Update available: v{update_info['latest_version']}")
                print(f"  Run 'tally update' to install")
    elif args.command == 'update':
        cmd_update(args)


if __name__ == '__main__':
    main()
