"""Bayesian posterior updates for an imperfect binary verifier."""


def posterior_error_probability(prior: float, sensitivity: float,
                                specificity: float, observation: str) -> float:
    for name, value in {"prior": prior, "sensitivity": sensitivity,
                        "specificity": specificity}.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    if observation.lower() == "pass":
        numerator = prior * (1.0 - sensitivity)
        denominator = numerator + (1.0 - prior) * specificity
    elif observation.lower() == "fail":
        numerator = prior * sensitivity
        denominator = numerator + (1.0 - prior) * (1.0 - specificity)
    else:
        raise ValueError("observation must be 'pass' or 'fail'")

    if denominator == 0:
        raise ZeroDivisionError("Posterior is undefined for a zero-probability observation")
    return numerator / denominator
