from conftest import MockEngine

from reasoning_reliability.strategies import DirectAnsweringStrategy
from reasoning_reliability.utils.presentation import chatbot_response, technical_payload


def test_raw_unparseable_response_is_still_readable(math_example):
    result = DirectAnsweringStrategy(MockEngine(["2 + 3 = 5"])).run(math_example)
    assert chatbot_response(result) == "2 + 3 = 5"
    payload = technical_payload(result)
    assert payload["parser_status"] == "strict_delimiter_not_found"
    assert payload["raw_response"] == "2 + 3 = 5"
    assert "difficulty" not in payload


def test_parsed_response_is_formatted_as_answer(math_example):
    result = DirectAnsweringStrategy(MockEngine(["FINAL_ANSWER: 5"])).run(math_example)
    assert chatbot_response(result) == "**Final answer:** 5"
    assert technical_payload(result)["parsed_answer"] == "5"
