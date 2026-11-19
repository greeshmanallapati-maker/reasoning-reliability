from reasoning_reliability.data.loader import deterministic_sample, preprocess_arc, preprocess_gsm8k


def test_gsm8k_preprocessing():
    result = preprocess_gsm8k({"question": "q", "answer": "x #### 42"}, "test", 0)
    assert result.answer == "42" and result.id == "gsm8k:0"


def test_arc_numeric_label_normalization():
    row = {"id": "x", "question": "q", "choices": {"label": ["1", "2"], "text": ["x", "y"]}, "answerKey": "2"}
    result = preprocess_arc(row, "ARC-Easy", "test", 0)
    assert result.answer == "2" and result.choices == {"1": "x", "2": "y"}


def test_deterministic_sampling(math_example):
    examples = [math_example.__class__(f"id:{i}", "gsm8k", "test", str(i), str(i)) for i in range(20)]
    assert [x.id for x in deterministic_sample(examples, 5, 7)] == [x.id for x in deterministic_sample(examples, 5, 7)]


def test_duplicate_ids_rejected(math_example):
    try:
        deterministic_sample([math_example, math_example], 1, 1)
        assert False
    except ValueError as error:
        assert "Duplicate" in str(error)

