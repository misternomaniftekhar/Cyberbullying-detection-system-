#!/usr/bin/env python3
"""Adds the v7 advanced tools (Drift & Slang Radar, Privacy Shield, Escalation Forecaster) to your Streamlit app.

    python apply_v7.py                      # patches ./streamlit_app.py
    python apply_v7.py path/to/app.py       # patches another file

Keep v7_addon.py in the same folder as this script. The patch only APPENDS a block to the end of your file; nothing
you already have is edited. It checks that your file has everything the add-on needs, writes a timestamped backup
first, and refuses to run twice.
"""
import pathlib
import re
import shutil
import sys
import time

BEGIN, END = "# >>> v7-addon-begin", "# <<< v7-addon-end"
REQUIRED = ["def judge_core", "def judge_many", "def strict_cfg", "def llm_json", "def build_context", "def clean_text",
            "def norm_for_sim", "def _skel", "def target_of", "def lexicon_hits", "def note_stats", "def badge", "def db",
            "def wilson", "def _scrub", "def ctx_html", "THREADS =", "LABELS =", "SEV_W =", "COLOR ="]
DOC_NOTE = """What's new in v7 (advanced tools)
    * Drift & Emerging-Slang Radar: PSI / KS / z-test drift monitor, keyword-filter blind rate, and a radar for new slang (spelling-merged)
    * Privacy Shield: Pakistan-format PII (CNIC, mobile, IBAN, card, email, Urdu numerals) redacted before the LLM, doxxing flag,
      log scrubbing, retention purge, pseudonymised research export
    * Escalation Forecaster: early warning from the start of a conversation, backtested against hidden last messages

"""


def main():
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "streamlit_app.py")
    addon_path = pathlib.Path(__file__).resolve().with_name("v7_addon.py")
    if not target.is_file():
        sys.exit(f"Cannot find {target}. Run this from your project folder or pass the path to your app.")
    if not addon_path.is_file():
        sys.exit(f"Cannot find {addon_path.name} next to this script.")
    src = target.read_text(encoding="utf-8")
    if BEGIN in src:
        sys.exit("The v7 add-on is already in this file. Nothing to do.")
    missing = [r for r in REQUIRED if not re.search(r"(?m)^" + re.escape(r), src)]
    if "CREATE VIEW v_effective" not in src:
        missing.append("CREATE VIEW v_effective")
    if missing:
        sys.exit("Your file is missing things the add-on depends on (it needs the v6 app):\n  - " + "\n  - ".join(missing)
                 + "\nNothing was changed.")
    addon = addon_path.read_text(encoding="utf-8")
    new_src = src.rstrip("\n") + "\n" + addon
    marker = "What's new in v6"
    if marker in new_src:
        new_src = new_src.replace(marker, DOC_NOTE + marker, 1)
    compile(new_src, str(target), "exec")  # syntax check before touching the disk
    backup = target.with_name(f"{target.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(target, backup)
    target.write_text(new_src, encoding="utf-8")
    print(f"Done. Added the v7 tools to {target} (backup: {backup.name}).")
    print("Restart the app: streamlit run", target.name)


if __name__ == "__main__":
    main()
