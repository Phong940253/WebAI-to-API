"""Clear stored Gemini credentials so the server can run in guest mode.

Counterpart to verify_login.py: that script saves the login state, this one
removes it so the runtime falls back to an anonymous guest session
(`[Gemini] guest_mode = true`, only `gemini-3-flash-lite` usable).

What it clears:
  1. The canonical login state (runtime/auth/gemini.json), backed up first.
  2. Cached Gemini WebAPI cookies (gemini-webapi cookie cache).
  3. Cookie values inside config.conf ([Gemini] and legacy [Cookies] keys),
     backed up first (config.conf.<timestamp>.bak; the first backup is kept).
  4. Enables [Gemini] guest_mode = true in config.conf.

Usage:
  python clear_auth.py            # switch to guest mode (backs up what it clears)
  python clear_auth.py --restore  # restore login state and config.conf backup
  python clear_auth.py --dry-run  # show planned changes without writing
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add src to sys.path to allow imports
sys.path.append(os.path.join(os.getcwd(), "src"))

from app.config import CONFIG  # noqa: E402

CONFIG_FILE = "config.conf"
CONFIG_BACKUP_GLOB = "config.conf.*.bak"
AUTH_STATE_FILENAME = "gemini.json"
COOKIE_CACHE_PREFIX = ".cached_cookies_"
COOKIE_OPTION_RE = re.compile(
    r"^(\s*(?:__Secure-1PSID(?:TS)?|gemini_cookie_1psid(?:ts)?)\s*=\s*)(.*)$"
)
SECTION_RE = re.compile(r"^\s*\[(?P<section>[^\]]+)\]\s*$")
GUEST_MODE_RE = re.compile(r"^(\s*guest_mode\s*=\s*)(.*)$")
CLEARED_SECTIONS = {"Gemini", "Cookies"}


def _auth_state_path() -> Path:
    auth_state_dir = CONFIG["Playwright"].get("auth_state_dir", "runtime/auth")
    return Path(auth_state_dir) / AUTH_STATE_FILENAME


def _cookie_cache_dir() -> Path:
    configured = os.environ.get("GEMINI_COOKIE_PATH")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "gemini_webapi"


def clear_login_state(*, dry_run: bool) -> list[str]:
    """Move gemini.json aside (timestamped backup). Returns report lines."""
    state_path = _auth_state_path()
    if not state_path.exists():
        return [f"Login state: nothing to clear ({state_path} not found)."]

    backup_path = state_path.with_name(f"{AUTH_STATE_FILENAME}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    if dry_run:
        return [f"Login state: would back up {state_path} -> {backup_path}."]

    state_path.rename(backup_path)
    return [f"Login state: backed up {state_path} -> {backup_path}."]


def restore_login_state() -> list[str]:
    """Restore the newest gemini.json backup. Returns report lines."""
    state_path = _auth_state_path()
    if state_path.exists():
        return [
            f"[WARN] {state_path} already exists; keeping the current login state."
        ]

    backups = sorted(
        state_path.parent.glob(f"{AUTH_STATE_FILENAME}.*.bak"),
        key=lambda path: path.stat().st_mtime,
    )
    if not backups:
        return [f"[WARN] No login backup found in {state_path.parent}; login state left unchanged."]

    backup_path = backups[-1]
    backup_path.rename(state_path)
    return [f"Login state: restored {backup_path} -> {state_path}."]


def clear_cookie_cache(*, dry_run: bool) -> list[str]:
    """Remove cached Gemini WebAPI cookies. Returns report lines."""
    cache_dir = _cookie_cache_dir()
    cache_files = sorted(cache_dir.glob(f"{COOKIE_CACHE_PREFIX}*.json"))
    if not cache_files:
        return [f"Cookie cache: nothing to clear ({cache_dir} has no cached cookies)."]

    if dry_run:
        return [f"Cookie cache: would remove {len(cache_files)} file(s) from {cache_dir}."]

    removed = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            removed += 1
        except OSError as error:
            print(f"[WARN] Could not remove {cache_file}: {error}")
    return [f"Cookie cache: removed {removed} file(s) from {cache_dir}."]


def rewrite_config(*, dry_run: bool) -> list[str]:
    """Blank config cookie values and enable guest_mode, preserving comments."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        if dry_run:
            return [f"Config: would create {config_path} with [Gemini] guest_mode = true."]
        config_path.write_text(
            "# Created by clear_auth.py: anonymous guest mode.\n"
            "[Gemini]\n"
            "guest_mode = true\n",
            encoding="utf-8",
        )
        return [f"Config: created {config_path} with [Gemini] guest_mode = true."]

    original_lines = config_path.read_text(encoding="utf-8").splitlines()
    report: list[str] = []
    new_lines: list[str] = []
    current_section = None
    gemini_header_index = None
    guest_mode_written = False

    for line in original_lines:
        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group("section")
            if current_section == "Gemini":
                gemini_header_index = len(new_lines)
            new_lines.append(line)
            continue

        if current_section in CLEARED_SECTIONS:
            cookie_match = COOKIE_OPTION_RE.match(line)
            if cookie_match and cookie_match.group(2).strip():
                report.append(
                    f"Config: cleared {current_section}.{cookie_match.group(1).split('=')[0].strip()} in {config_path}."
                )
                line = f"{cookie_match.group(1)}"

        if current_section == "Gemini":
            guest_match = GUEST_MODE_RE.match(line)
            if guest_match:
                report.append(
                    f"Config: set [Gemini] guest_mode = true (was '{guest_match.group(2).strip()}')."
                )
                line = f"{guest_match.group(1)}true"
                guest_mode_written = True

        new_lines.append(line)

    if not guest_mode_written:
        if gemini_header_index is not None:
            new_lines.insert(gemini_header_index + 1, "guest_mode = true")
        else:
            new_lines.extend(["", "[Gemini]", "guest_mode = true"])
        report.append(f"Config: added [Gemini] guest_mode = true to {config_path}.")

    if not report:
        return [f"Config: {config_path} already has empty cookies and guest_mode = true."]

    if dry_run:
        return [f"[DRY-RUN] {entry}" for entry in report] + [
            f"[DRY-RUN] Config: would back up {config_path} before writing "
            "(only if no backup exists yet)."
        ]

    report.extend(_backup_config(config_path))
    config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return report


def _backup_config(config_path: Path) -> list[str]:
    """Copy config.conf aside before the first modification. First backup wins."""
    existing = sorted(
        config_path.parent.glob(CONFIG_BACKUP_GLOB),
        key=lambda path: path.stat().st_mtime,
    )
    if existing:
        return [
            f"Config: kept existing backup {existing[-1]} (first backup is preserved)."
        ]

    backup_path = config_path.with_name(
        f"config.conf.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    )
    shutil.copy2(config_path, backup_path)
    return [f"Config: backed up {config_path} -> {backup_path}."]


def restore_config() -> list[str]:
    """Restore the newest config.conf backup over config.conf (backup is consumed)."""
    config_path = Path(CONFIG_FILE)
    backups = sorted(
        config_path.parent.glob(CONFIG_BACKUP_GLOB),
        key=lambda path: path.stat().st_mtime,
    )
    if not backups:
        return ["Config: no config.conf backup found; config left unchanged."]

    backup_path = backups[-1]
    backup_path.replace(config_path)
    return [f"Config: restored {backup_path} -> {config_path} (backup consumed)."]


def print_report(report: list[str], *, dry_run: bool) -> None:
    title = "GUEST MODE AUTH CLEAR"
    if dry_run:
        title += " (dry run)"
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for entry in report:
        print(f"  - {entry}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore the gemini.json login backup and the config.conf backup",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change without modifying anything",
    )
    args = parser.parse_args()

    if args.restore:
        for entry in restore_login_state() + restore_config():
            print(f"  - {entry}")
        print(
            "\nLogin state and config.conf restored. Restart the server to apply, e.g.:"
            "\n  docker compose restart web_ai"
        )
        return 0

    report: list[str] = []
    report.extend(clear_login_state(dry_run=args.dry_run))
    report.extend(clear_cookie_cache(dry_run=args.dry_run))
    report.extend(rewrite_config(dry_run=args.dry_run))
    print_report(report, dry_run=args.dry_run)

    if args.dry_run:
        print("\nNo changes were made (dry run).")
        return 0

    print("\nNEXT STEPS:")
    print("  1. Restart the server so the client re-initializes in guest mode, e.g.:")
    print("       docker compose restart web_ai")
    print("     or: poetry run python src/run.py")
    print("  2. Verify: GET /v1/auth/status should report GUEST, and")
    print("     POST /v1/chat/completions with model 'gemini-3-flash-lite' should work.")
    print("  3. Other models return HTTP 400; store/conversation_id (persistent chat)")
    print("     returns HTTP 401 until you sign in again.")
    print("  4. Undo with 'python clear_auth.py --restore' or re-login with")
    print("     'poetry run python verify_login.py'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
