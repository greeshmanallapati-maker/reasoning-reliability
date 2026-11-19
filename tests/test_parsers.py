import pytest

from reasoning_reliability.data.answer_parser import (
    normalize_arc_label,
    normalize_number,
    parse_arc_prediction,
    parse_gsm8k_prediction,
    parse_gsm8k_reference,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$1,234.50", "1234.5"), ("-02.00", "-2"), ("0", "0"), ("bad", None)],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_gsm8k_reference_and_prediction():
    assert parse_gsm8k_reference("work\n#### 1,250") == "1250"
    assert parse_gsm8k_prediction("steps\nFINAL_ANSWER: $1,250.00") == "1250"
    assert parse_gsm8k_prediction("<final_answer>42</final_answer>") == "42"


def test_prediction_requires_delimiter():
    assert parse_gsm8k_prediction("the answer is 5") is None


def test_arc_labels_and_prediction():
    assert normalize_arc_label("2", ["A", "B", "C"]) == "B"
    assert parse_arc_prediction("FINAL_ANSWER: (C)", ["A", "B", "C"]) == "C"
    assert parse_arc_prediction("<final_answer>B</final_answer>", ["A", "B", "C"]) == "B"
    assert parse_arc_prediction("FINAL_ANSWER: Z", ["A", "B"]) is None

