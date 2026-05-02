from eval.metrics import exact_match, normalized_match, token_reduction_pct
from eval.slicing import slice_results, length_bucket


def test_exact_match_pass():
    assert exact_match("hello\n", "hello\n")


def test_exact_match_fail_on_whitespace():
    assert not exact_match("hello \n", "hello\n")


def test_normalized_match_collapses_whitespace():
    assert normalized_match("hello   world\n", "hello world\n")


def test_normalized_match_strips_trailing_ws():
    assert normalized_match("hello   \nworld   \n", "hello\nworld\n")


def test_normalized_match_still_fails_on_content_diff():
    assert not normalized_match("hello world", "hello earth")


def test_token_reduction_positive():
    pct = token_reduction_pct(n_input=100, n_output=30)
    assert abs(pct - 0.7) < 1e-9


def test_token_reduction_zero_for_passthrough():
    assert token_reduction_pct(100, 100) == 0.0


def test_token_reduction_negative_when_output_longer():
    assert token_reduction_pct(100, 150) == -0.5


def test_length_bucket():
    assert length_bucket(100) == "<512"
    assert length_bucket(1024) == "512-2k"
    assert length_bucket(5000) == "2k-8k"


def test_slice_results_groups():
    records = [
        {"recipe": "passthrough", "passed": True, "metric": 1.0},
        {"recipe": "passthrough", "passed": True, "metric": 0.9},
        {"recipe": "noisy_logs", "passed": False, "metric": 0.0},
    ]
    out = slice_results(records, key_fn=lambda r: r["recipe"])
    assert "passthrough" in out
    assert out["passthrough"]["count"] == 2
    assert out["passthrough"]["pass_rate"] == 1.0
