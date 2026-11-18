import pytest
from dav.verifier import posterior_error_probability


def test_pass_reduces_risk_for_reasonable_verifier():
    posterior = posterior_error_probability(0.4, 0.9, 0.9, "pass")
    assert posterior < 0.4


def test_fail_increases_risk_for_reasonable_verifier():
    posterior = posterior_error_probability(0.2, 0.9, 0.9, "fail")
    assert posterior > 0.2


def test_invalid_observation_rejected():
    with pytest.raises(ValueError):
        posterior_error_probability(0.2, 0.9, 0.9, "unknown")
