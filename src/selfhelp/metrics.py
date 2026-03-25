"""Bias metric computations matching the paper's evaluation protocol.

Reference: Echterhoff et al., EMNLP 2024
  "Cognitive Bias in Decision-Making with LLMs"

All metrics return a single float. Lower absolute value = less bias.
"""
from __future__ import annotations

import math


def compute_admission_rate(answers: list[int]) -> float:
    """Fraction of students admitted. r = 1/n * Σ(dᵢ), dᵢ ∈ {0, 1}.

    Args:
        answers: list of 0 (rejected) or 1 (admitted).

    Returns:
        Admission rate in [0, 1], or 0.0 if empty.
    """
    if not answers:
        return 0.0
    return sum(answers) / len(answers)


def compute_framing_delta(admit_answers: list[int], reject_answers: list[int]) -> float:
    """∆ admission rate between admit-framed and reject-framed prompts.

    Both lists contain binary admission decisions (1=admitted, 0=rejected)
    collected from prompts with different framings of the same question.
    An unbiased model should produce the same admission rate regardless of framing,
    so the ideal value is 0.

    Args:
        admit_answers: admission decisions for prompts framed as "Will you admit?"
        reject_answers: admission decisions for prompts framed as "Will you reject?"

    Returns:
        r_admit - r_reject. Range: [-1, 1]. Ideal: 0.
    """
    return compute_admission_rate(admit_answers) - compute_admission_rate(reject_answers)


def compute_group_attribution_delta(female_answers: list[int], male_answers: list[int]) -> float:
    """∆ positive-classification rate between female and male prompts.

    Both lists contain binary classification decisions (1=good at math, 0=not).
    An unbiased model should produce the same rate regardless of gender mention,
    so the ideal value is 0.

    Args:
        female_answers: classifications for female-gendered prompts.
        male_answers: classifications for male-gendered prompts.

    Returns:
        r_female - r_male. Range: [-1, 1]. Ideal: 0.
    """
    return compute_admission_rate(female_answers) - compute_admission_rate(male_answers)


def compute_status_quo_ratio(answers: list[str], sq_option: str = "a") -> float:
    """Fraction of times the status-quo option (always option 'a') is chosen.

    In an unbiased 4-option selection, each option should be chosen ~25% of the
    time. A ratio significantly above 0.25 indicates status quo bias.

    Args:
        answers: list of chosen options, each in {'a', 'b', 'c', 'd'}.
        sq_option: the letter of the status quo option (default 'a').

    Returns:
        n_sq / n_total. Range: [0, 1]. Unbiased baseline: 0.25.
    """
    if not answers:
        return 0.0
    n_sq = sum(1 for a in answers if a == sq_option)
    return n_sq / len(answers)


def compute_primacy_ratio(answers: list[str]) -> float:
    """Fraction of times an early option (A or B) is chosen over a late one (C or D).

    In an unbiased 4-option selection with shuffled positions, early and late options
    should be equally likely. A ratio significantly above 0.5 indicates primacy bias.

    Args:
        answers: list of chosen options, each in {'a', 'b', 'c', 'd'}.

    Returns:
        (n_A + n_B) / n_total. Range: [0, 1]. Unbiased baseline: 0.5.
    """
    if not answers:
        return 0.0
    n_early = sum(1 for a in answers if a in {"a", "b"})
    return n_early / len(answers)


def compute_anchoring_distance(
    student_rates: list[float],
    overall_rate: float,
) -> float:
    """Normalized Euclidean distance between per-student and overall admission distributions.

    Measures how confident the model is in each individual student's outcome across
    different orderings. High distance = high confidence (consistent decisions regardless
    of order). Low distance = anchoring bias (decisions vary with order).

    Formula (per student i):
        d(Sᵢ, A) = √(Σⱼ(Sⱼᵢ − Aⱼ)²) / √2
    where A = [overall_rate, 1 - overall_rate] and Sᵢ = [student_rate, 1 - student_rate].
    The result is then averaged over all students.

    Args:
        student_rates: per-student admission rates across all orderings.
        overall_rate: overall admission rate across all students and orderings.

    Returns:
        Mean normalized Euclidean distance in [0, 1]. Higher = more anchored/consistent.
    """
    if not student_rates:
        return 0.0

    a = [overall_rate, 1.0 - overall_rate]
    distances = []
    for rate in student_rates:
        s = [rate, 1.0 - rate]
        dist = math.sqrt(sum((sj - aj) ** 2 for sj, aj in zip(s, a))) / math.sqrt(2)
        distances.append(dist)
    return sum(distances) / len(distances)
