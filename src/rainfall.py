"""
Rainfall, crop suitability, and regional similarity.

Contains:
    - Region: monthly rainfall series (12 values in mm)
    - CropRule: monthly rainfall suitability range for a crop
    - cosine_similarity: proper NumPy implementation (not math.cos)
    - cosine_similarity_scipy: cross-check against scipy.spatial
    - pearson, euclidean: complementary similarity/distance measures
    - similarity_matrix: pairwise matrix over a list of Regions
    - detect_rainy_seasons, classify_seasonality: peak detection
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine


# ============================================================
# Region
# ============================================================
class Region:
    """Holds a 12-element monthly rainfall series (mm) for one region."""

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def __init__(self, name: str, rainfall: np.ndarray) -> None:
        rainfall = np.asarray(rainfall, dtype=float)
        if rainfall.shape != (12,):
            raise ValueError("rainfall must be a length-12 array.")
        if np.any(rainfall < 0):
            raise ValueError("rainfall cannot be negative.")
        self.name = name
        self.rainfall = rainfall

    def __len__(self) -> int:
        return 12

    def __repr__(self) -> str:
        return (f"Region(name={self.name!r}, "
                f"annual_total={self.annual_total():.0f} mm)")

    def annual_total(self) -> float:
        """Sum of monthly rainfall (mm)."""
        return float(self.rainfall.sum())

    def mean(self) -> float:
        """Mean monthly rainfall (mm)."""
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        """Name of the month with highest rainfall."""
        return self.MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        """Name of the month with lowest rainfall."""
        return self.MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        """Coefficient of variation of the monthly rainfall."""
        m = self.mean()
        return float(self.rainfall.std(ddof=1) / m) if m else float("nan")


# ============================================================
# Crop suitability rules
# ============================================================
@dataclass
class CropRule:
    """
    Monthly rainfall suitability range (mm) for a crop.

    `low_mm` and `high_mm` are the lower and upper bounds of the
    favourable monthly rainfall range. Below `low_mm` is drought risk,
    above `high_mm` is waterlogging risk.
    """
    name: str
    low_mm: float
    high_mm: float

    def classify(self, mm: float) -> str:
        """Return a suitability label for a given monthly rainfall."""
        if mm < self.low_mm:
            return f"Drought risk for {self.name}"
        if mm > self.high_mm:
            return f"Waterlogging risk for {self.name}"
        return f"Good for {self.name}"


def default_crop_rules() -> list[CropRule]:
    """
    Return a list of the three default crops with illustrative thresholds.

    Thresholds are approximate monthly ranges adapted from FAO crop
    water-requirement tables and Ugandan agronomic guidance. They are
    illustrative, not policy-grade.
    """
    return [
        CropRule("maize",  low_mm=80.0,  high_mm=200.0),
        CropRule("beans",  low_mm=60.0,  high_mm=180.0),
        CropRule("coffee", low_mm=100.0, high_mm=220.0),
    ]


# ============================================================
# Similarity / distance measures
# ============================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Proper cosine similarity: dot(a, b) / (||a|| * ||b||).

    NOTE: this is NOT math.cos(). math.cos(x) computes the cosine of
    an angle x; cosine similarity measures the angle between two
    vectors and returns a value in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def cosine_similarity_scipy(a: np.ndarray, b: np.ndarray) -> float:
    """Cross-check via 1 - scipy.spatial.distance.cosine(a, b)."""
    return float(1.0 - cosine(np.asarray(a, float), np.asarray(b, float)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two rainfall series."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def similarity_matrix(regions: list[Region], metric=cosine_similarity) -> np.ndarray:
    """Symmetric matrix of pairwise metric values across regions."""
    n = len(regions)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = metric(regions[i].rainfall, regions[j].rainfall)
            M[i, j] = M[j, i] = v
    return M


# ============================================================
# Rainy season detection
# ============================================================
def detect_rainy_seasons(
    region: Region,
    min_prominence: float = 10.0,
) -> list[int]:
    """
    Return 1-based month indices of rainfall peaks.

    Uses scipy.signal.find_peaks with a minimum prominence to avoid
    detecting noise.
    """
    peaks, _ = find_peaks(region.rainfall, prominence=min_prominence)
    return [int(p) + 1 for p in peaks]


def classify_seasonality(
    region: Region,
    min_prominence: float = 10.0,
) -> str:
    """Return 'unimodal', 'bimodal', or 'no clear peak'."""
    peaks = detect_rainy_seasons(region, min_prominence)
    if len(peaks) >= 2:
        return "bimodal"
    if len(peaks) == 1:
        return "unimodal"
    return "no clear peak"


# ============================================================
# Synthetic data generator
# ============================================================
def synthetic_uganda_regions() -> list[Region]:
    """
    Return five illustrative Ugandan regions with realistic monthly
    rainfall patterns.

    Kampala / Entebbe / Mbarara: bimodal (two rainy seasons)
    Gulu / Soroti: unimodal (one long rainy season)
    """
    return [
        Region("Kampala", np.array(
            [60, 80, 130, 180, 160, 60, 40, 70, 130, 170, 140, 80], dtype=float)),
        Region("Entebbe", np.array(
            [70, 90, 140, 200, 170, 70, 50, 80, 140, 180, 150, 90], dtype=float)),
        Region("Gulu", np.array(
            [20, 30, 70, 130, 160, 150, 140, 160, 150, 120, 60, 25], dtype=float)),
        Region("Mbarara", np.array(
            [50, 60, 100, 140, 110, 40, 30, 60, 110, 140, 120, 70], dtype=float)),
        Region("Soroti", np.array(
            [25, 35, 80, 120, 140, 130, 120, 140, 130, 100, 55, 30], dtype=float)),
    ]