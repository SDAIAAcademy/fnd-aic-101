# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Scan what is about to be committed for what must never enter the history of a public course:
identifiers with a valid shape (Saudi national IDs, mobile numbers, IBANs, card numbers that pass
the Luhn check) and the names of real brands, products, apps, card schemes and places.

From the course root, before every commit:

    uv run labs/_tools/scan_staged.py            # the staged files (what git commit will record)
    uv run labs/_tools/scan_staged.py --all      # every tracked and untracked file of the course

It reads the staged version of each file (git show :path), notebooks included, prints every hit
with its file and exits 1 when there is one. The case of this course has no phone numbers, no
national IDs, no IBANs and no card numbers at all: clients are codes such as C0042, staff are
roles, and the city and its sectors are unnamed. So any hit is worth a look.

Allowed on purpose, and therefore not in the list below: the tool-slot names of the catalog
(JupyterLab, scikit-learn, pandas, Matplotlib, Polars, seaborn, tidymodels, NumPy, uv) and the
free hosted notebook service (Colab), which the house rules name as the place labs run.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELF = "labs/_tools/scan_staged.py"
TEXT = {".html", ".md", ".json", ".jsonl", ".py", ".txt", ".csv", ".toml", ".yml", ".yaml",
        ".mjs", ".js", ".css", ".ipynb", ".gitignore", ".gitattributes", ".lock", ""}

PATTERNS = {
    "national ID shape (10 digits from 1 or 2)": re.compile(r"(?<![\d.])[12]\d{9}(?![\d])"),
    "mobile shape (05 and 8 digits)": re.compile(r"(?<![\d.])(?:\+?966\s?|0)5\d{8}(?!\d)"),
    "IBAN shape": re.compile(r"\bSA\d{2}\s?(?:[0-9A-Z]{4}\s?){5}\b"),
}
ASCII_NAMES = [
    # messaging, social and government apps
    "WhatsApp", "Telegram", "Snapchat", "Instagram", "TikTok", "Absher", "Nafath", "Tawakkalna",
    "Sehhaty", "Mawid", "Balady", "Etimad", "Ejar", "Sakani",
    # card schemes and payments
    "Visa", "Mastercard", "MasterCard", "Amex", "American Express", "Apple Pay", "STC Pay", "SADAD",
    # AI products and providers, cloud and office software
    "OpenAI", "ChatGPT", "GPT-4", "GPT-5", "Gemini", "Claude", "Anthropic", "Copilot", "Llama",
    "Mistral", "DeepSeek", "Qwen", "Azure", "AWS", "Amazon", "Microsoft", "Excel", "Salesforce",
    "Oracle", "SAP", "Fitbit", "Garmin",
    # Saudi cities, regions and landmarks
    "Riyadh", "Jeddah", "Jiddah", "Dammam", "Khobar", "Dhahran", "Makkah", "Mecca", "Madinah",
    "Medina", "Taif", "Abha", "Tabuk", "Hail", "Qassim", "Buraidah", "Unaizah", "Jazan", "Jizan",
    "Najran", "Al-Ahsa", "Hofuf", "Jubail", "Yanbu", "AlUla", "NEOM", "Qatif", "Khamis Mushait",
    "Asir", "Al-Baha", "Al-Jouf", "Sakaka", "Arar",
    # real Saudi health providers and bodies a home-care case could be mistaken for
    "Ministry of Health", "MOH", "SFDA", "CHI", "Saudi German", "Fakeeh", "Almoosa", "Dallah",
    "Mouwasat", "Sulaiman Al Habib", "Andalusia", "Future Care",
]
ARABIC_NAMES = [
    "واتساب", "تيليجرام", "سناب شات", "أبشر", "نفاذ", "توكلنا", "صحتي", "فيزا", "ماستركارد",
    "الرياض", "جدة", "الدمام", "الخبر", "مكة", "المدينة المنورة", "الطائف", "أبها", "تبوك",
    "حائل", "القصيم", "بريدة", "جازان", "نجران", "الأحساء", "الجبيل", "ينبع", "العلا", "نيوم",
    "وزارة الصحة",
]


def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def files(all_files: bool) -> list[tuple[str, str]]:
    if all_files:
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
        return [(n, (ROOT / n).read_text(encoding="utf-8", errors="replace")) for n in names
                if n and (ROOT / n).is_file()]
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")
    return [(n, git("show", f":{n}").decode("utf-8", errors="replace")) for n in names if n]


def scan(name: str, text: str) -> list[str]:
    hits = []
    for what, rx in PATTERNS.items():
        hits += [f"{name}: {what}: {m.group(0)}" for m in rx.finditer(text)]
    for m in re.finditer(r"(?<![\d.])\d{13,19}(?![\d])", text):
        if luhn(m.group(0)):
            hits.append(f"{name}: card-shaped number (passes Luhn): {m.group(0)}")
    for word in ASCII_NAMES:
        if re.search(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", text):
            hits.append(f"{name}: brand, product or place: {word}")
    for word in ARABIC_NAMES:
        if re.search(r"(?<![؀-ۿ])" + re.escape(word) + r"(?![؀-ۿ])", text):
            hits.append(f"{name}: brand, product or place: {word}")
    return hits


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    all_files = "--all" in sys.argv[1:]
    hits, count = [], 0
    for name, text in files(all_files):
        if name == SELF or Path(name).suffix.lower() not in TEXT:
            continue
        count += 1
        hits += scan(name, text)
    for h in hits:
        print(h)
    print(f"scan: {count} {'files' if all_files else 'staged files'}, {len(hits)} hits")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
