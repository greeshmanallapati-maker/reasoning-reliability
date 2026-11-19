from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

FINAL_RE = re.compile(r"FINAL_ANSWER\s*(?::|=|\bis\b)\s*(.+)", re.IGNORECASE)
FINAL_XML_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?(?:[$£€₹]\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def normalize_number(value: str | int | float | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    text = re.sub(r"^[\s$£€₹]+", "", text)
    text = text.rstrip(". ")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def parse_gsm8k_reference(answer: str) -> str | None:
    marker = answer.rsplit("####", 1)
    candidate = marker[-1] if len(marker) == 2 else answer
    matches = NUMBER_RE.findall(candidate)
    return normalize_number(matches[-1]) if matches else None


def parse_gsm8k_prediction(text: str) -> str | None:
    matches = FINAL_RE.findall(text or "")
    if not matches:
        matches = FINAL_XML_RE.findall(text or "")
    if not matches:
        return None
    numbers = NUMBER_RE.findall(matches[-1].splitlines()[0])
    return normalize_number(numbers[0]) if numbers else None


def normalize_arc_label(label: str | int | None, labels: list[str] | None = None) -> str | None:
    if label is None:
        return None
    text = str(label).strip().upper().strip("().: ")
    valid = [str(x).upper() for x in (labels or ["A", "B", "C", "D", "E"])]
    if text in valid:
        return text
    if text.isdigit():
        index = int(text) - 1
        return valid[index] if 0 <= index < len(valid) else None
    return None


def parse_arc_prediction(text: str, labels: list[str]) -> str | None:
    matches = FINAL_RE.findall(text or "")
    if not matches:
        matches = FINAL_XML_RE.findall(text or "")
    if not matches:
        return None
    token = re.split(r"\s+", matches[-1].strip())[0]
    return normalize_arc_label(token, labels)


def split_reasoning(text: str) -> tuple[str, str]:
    match = FINAL_RE.search(text or "")
    if not match:
        return (text.strip(), "")
    return (text[: match.start()].strip(), match.group(1).splitlines()[0].strip())

