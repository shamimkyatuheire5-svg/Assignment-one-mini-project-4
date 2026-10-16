"""
Unit tests for src/rainfall.py.

Covers:
    - normal cases (region stats, cosine similarity, seasonal detection)
    - edge cases (wrong shape, negative rainfall, zero vector)
    - the mandatory cross-check against scipy.spatial.distance.cosine
"""
import numpy as np
import pytest

from src.rainfall import (
    Region,
    CropRule,
    default_crop_rules,
    cosine_similarity,
    cosine_similarity_scipy,
    pearson,
    euclidean,
    similarity_matrix,
    detect_rainy_seasons,
    classify_seasonality,
    synthetic_uganda_regions,
)


MONTHLY = np.array([60, 80, 130, 180, 160, 60, 40, 70, 130, 170, 140, 80], dtype=float)


# ------------------------------------------------------------
# Region
# ------------------------------------------------------------
def test_region_annual_total_hand_check():
    """Sum of monthly values should match by hand."""
    r = Region("Kampala", MONTHLY)
    assert r.annual_total() == pytest.approx(1300.0)


def test_region_mean_and_extremes():
    r = Region("Kampala", MONTHLY)
    assert r.mean() == pytest.approx(1300.0 / 12)
    assert r.wettest_month() == "Apr"
    assert r.driest_month() == "Jul"


def test_region_wrong_shape_raises():
    """Edge case: rainfall must have exactly 12 entries."""
    with pytest.raises(ValueError, match="length-12"):
        Region("Bad", np.array([1, 2, 3]))


def test_region_negative_raises():
    """Edge case: negative rainfall rejected."""
    with pytest.raises(ValueError, match="negative"):
        Region("Bad", np.array([-1.0] * 12))


def test_len_and_repr():
    r = Region("Kampala", MONTHLY)
    assert len(r) == 12
    assert "Kampala" in repr(r)
    assert "1300" in repr(r)


# ------------------------------------------------------------
# CropRule
# ------------------------------------------------------------
def test_crop_rule_classification():
    rule = CropRule("maize", low_mm=80.0, high_mm=200.0)
    assert rule.classify(50) == "Drought risk for maize"
    assert rule.classify(120) == "Good for maize"
    assert rule.classify(250) == "Waterlogging risk for maize"


def test_default_crop_rules_three_crops():
    names = [r.name for r in default_crop_rules()]
    assert names == ["maize", "beans", "coffee"]


# ------------------------------------------------------------
# Similarity measures
# ------------------------------------------------------------
def test_cosine_similarity_of_identical_vectors_is_one():
    """Two identical vectors have cosine similarity 1."""
    assert cosine_similarity(MONTHLY, MONTHLY) == pytest.approx(1.0)


def test_cosine_similarity_hand_check():
    """cos([1,0], [0,1]) = 0 (orthogonal)."""
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_matches_scipy():
    """Our NumPy implementation must agree with scipy.spatial.distance.cosine."""
    a = np.array([60.0, 80.0, 130.0, 180.0])
    b = np.array([70.0, 90.0, 140.0, 200.0])
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity_scipy(a, b), rel=1e-9)


def test_cosine_similarity_zero_vector():
    """Edge case: a zero vector has no direction -> NaN."""
    assert np.isnan(cosine_similarity(np.zeros(4), MONTHLY[:4]))


def test_pearson_perfect_correlation():
    """A vector is perfectly correlated with itself."""
    assert pearson(MONTHLY, MONTHLY) == pytest.approx(1.0)


def test_euclidean_zero_distance():
    """Two identical vectors have distance 0."""
    assert euclidean(MONTHLY, MONTHLY) == pytest.approx(0.0)


# ------------------------------------------------------------
# Similarity matrix
# ------------------------------------------------------------
def test_similarity_matrix_shape_and_diagonal():
    regions = synthetic_uganda_regions()
    M = similarity_matrix(regions, metric=cosine_similarity)
    assert M.shape == (5, 5)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)   # symmetric


# ------------------------------------------------------------
# Seasonality
# ------------------------------------------------------------
def test_kampala_is_bimodal():
    """Kampala has two rainy seasons."""
    regions = {r.name: r for r in synthetic_uganda_regions()}
    assert classify_seasonality(regions["Kampala"]) == "bimodal"

def test_gulu_detects_two_peaks():
    """Gulu's synthetic series has two wet peaks (a mid-season dip)."""
    regions = {r.name: r for r in synthetic_uganda_regions()}
    peaks = detect_rainy_seasons(regions["Gulu"])
    assert len(peaks) == 2

def test_kampala_has_two_peaks():
    """Kampala's bimodal pattern should also produce two peaks."""
    regions = {r.name: r for r in synthetic_uganda_regions()}
    peaks = detect_rainy_seasons(regions["Kampala"])
    assert len(peaks) >= 2

def test_detect_rainy_seasons_returns_1_based_months():
    regions = {r.name: r for r in synthetic_uganda_regions()}
    peaks = detect_rainy_seasons(regions["Kampala"])
    # All values must be within 1..12
    assert all(1 <= p <= 12 for p in peaks)