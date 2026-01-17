"""
Tally 'update' command - Check for and install updates.
"""

import sys

from .._version import (
    VERSION,
    get_latest_release_info,
    perform_update,
)

from ..cli_utils import resolve_config_dir
from ..migrations import run_migrations


def cmd_update(args):
    """Handle the update command."""
    if args.prerelease:
        print("Checking for development builds...")
    else:
        print("Checking for updates...")

    # Get release info (may fail if offline or rate-limited)
    release_info = get_latest_release_info(prerelease=args.prerelease)
    has_update = False
    is_prerelease_to_stable = False

    if release_info:
        latest = release_info['version']
        current = VERSION

        # Show version comparison
        from .._version import _version_greater
        has_update = _version_greater(latest, current)

        # Special case: user is on prerelease (-dev) and checking for stable
        # Offer to switch to stable even if stable has lower base version
        if not args.prerelease and '-dev' in current and '-dev' not in latest:
            is_prerelease_to_stable = True
            has_update = True  # Always offer stable when on prerelease

        if has_update:
            if args.prerelease:
                print(f"Development build available: v{latest} (current: v{current})")
            elif is_prerelease_to_stable:
                print(f"Stable release available: v{latest} (current: v{current})")
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
    config_dir = resolve_config_dir(args, required=False)
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

    # Perform binary update (force=True when switching from prerelease to stable)
    print()
    success, message = perform_update(release_info, force=is_prerelease_to_stable)

    if success:
        print(f"\n✓ {message}")

        # Show release notes for stable releases only
        if not args.prerelease and release_info:
            _show_release_notes(release_info)
    else:
        print(f"\n✗ {message}")
        sys.exit(1)


def _render_markdown_line(line: str) -> str:
    """Render a single markdown line with ANSI formatting."""
    import re

    # ANSI codes
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    stripped = line.strip()

    # Headers: ### Text -> bold
    if stripped.startswith('#'):
        text = stripped.lstrip('#').strip()
        return f"{BOLD}{text}{RESET}"

    # Bullet points: - or * -> •
    if stripped.startswith(('- ', '* ')):
        text = stripped[2:]
        # Apply inline formatting to bullet text
        text = re.sub(r'\*\*(.+?)\*\*', rf'{BOLD}\1{RESET}', text)
        text = re.sub(r'`(.+?)`', rf'{DIM}\1{RESET}', text)
        return f"  • {text}"

    # Inline bold: **text** -> bold
    line = re.sub(r'\*\*(.+?)\*\*', rf'{BOLD}\1{RESET}', line)

    # Inline code: `code` -> dim
    line = re.sub(r'`(.+?)`', rf'{DIM}\1{RESET}', line)

    return line


def _show_release_notes(release_info: dict) -> None:
    """Display a summary of release notes and link to full release."""
    body = release_info.get('body', '')
    release_url = release_info.get('release_url', '')

    if not body:
        if release_url:
            print(f"\nSee full release notes at: {release_url}")
        return

    # Truncate at install section (auto-appended by workflow)
    for marker in ('## Install', '### Install'):
        if marker in body:
            body = body.split(marker)[0]

    # Render markdown lines
    lines = body.strip().split('\n')
    rendered = []
    total_chars = 0
    max_chars = 600
    in_code_block = False

    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            # Indent code block content
            rendered.append(f"    {line}")
        else:
            rendered.append(_render_markdown_line(line))

        total_chars += len(line)
        if total_chars >= max_chars:
            rendered.append("...")
            break

    if rendered:
        print("\n--- What's New ---")
        print('\n'.join(rendered))

    if release_url:
        print(f"\nFull release notes: {release_url}")
